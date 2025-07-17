# src/qera_core/ues_model/transformer_gnn_core.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Any, List, Union, Tuple # Added Tuple for type hints

# Import the encoders. Access their static methods via the class.
# Note: CircuitEncoder's static method is *not* imported from here, it's defined below in this file.
from src.qera_core.ues_model.input_encoders import NoiseEncoder, QECCodeEncoder, CircuitEncoder


# --- DEBUG PRINT: Confirm which CircuitEncoder is imported ---
print(f"--- DEBUG: CircuitEncoder is imported from: {CircuitEncoder.__module__}") 

# --- Placeholder for a basic GNN layer ---
class SimpleGraphConv(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(output_dim, activation='relu')

    def call(self, inputs):
        return self.dense(inputs)

# --- Transformer Block (standard implementation) ---
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- NEW HELPER FUNCTION for CircuitEncoder preprocessing (MOVED HERE from input_encoders.py) ---
def _preprocess_raw_circuit_ops(
    circuit_ops_batch: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]], 
    max_gates: int, 
    gate_types: Dict[str, int]
) -> tf.Tensor:
    """
    Helper method to preprocess raw circuit_ops (list of dicts) into a padded tensor of gate IDs.
    This method is called by UESModel.call BEFORE passing to CircuitEncoder.call.
    """
    if not isinstance(circuit_ops_batch, list) or (circuit_ops_batch and not isinstance(circuit_ops_batch[0], list)):
        circuit_ops_batch = [circuit_ops_batch]

    batched_gate_ids = []
    for circuit_ops in circuit_ops_batch: 
        gate_ids_single_circuit = []
        for op in circuit_ops:
            gate_name = op.get('name', 'other')
            gate_id = gate_types.get(gate_name, gate_types['other'])
            gate_ids_single_circuit.append(gate_id)
        
        padded_gate_ids = gate_ids_single_circuit + [gate_types['other']] * (max_gates - len(gate_ids_single_circuit))
        padded_gate_ids = padded_gate_ids[:max_gates]
        
        batched_gate_ids.append(padded_gate_ids)
    
    return tf.constant(batched_gate_ids, dtype=tf.int32)


class UESModel(Model):
    """
    The Universal Error Symphony (UES) AI model.
    Combines inputs from noise, QEC code, and circuit to output QEC strategies.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.noise_encoder = NoiseEncoder(output_dim=config['noise_embed_dim'])
        self.qec_code_encoder = QECCodeEncoder(output_dim=config['qec_embed_dim'])
        self.circuit_encoder = CircuitEncoder(max_gates=config['max_circuit_gates'],
                                             gate_embedding_dim=config['gate_embed_dim'])
        
        self.gnn_placeholder = SimpleGraphConv(config['graph_embed_dim']) 
        
        total_combined_embed_dim = config['noise_embed_dim'] + \
                                   config['qec_embed_dim'] + \
                                   (config['gate_embed_dim'] * 2) + \
                                   config['graph_embed_dim'] 

        self.transformer_block = TransformerBlock(
            embed_dim=total_combined_embed_dim, 
            num_heads=config['transformer_num_heads'],
            ff_dim=config['transformer_ff_dim']
        )

        self.fusion_layer = layers.Dense(config['fusion_dim'], activation='relu')

        self.mitigation_output = layers.Dense(config['num_mitigation_actions'], activation='softmax')
        self.decoder_output = layers.Dense(config['num_decoder_choices'], activation='softmax')
        self.compiler_output = layers.Dense(config['num_compiler_actions'], activation='softmax')

    def call(self, inputs: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]], List[List[Dict[str, Any]]]]], training: bool = False):
        # Inputs: {'noise_data': dict or list of dicts,
        #          'qec_code_props': dict or list of dicts,
        #          'circuit_ops': list of dicts or list of lists of dicts}
        
        # --- CRITICAL FIX: Pre-process raw inputs using static methods ---
        # Call the static preprocessing methods BEFORE passing to the encoder layers.
        
        # NoiseEncoder input preprocessing (using its static method from input_encoders.py)
        noise_embed = self.noise_encoder(
            NoiseEncoder._preprocess_raw_noise_data(inputs['noise_data'], self.noise_encoder.expected_noise_keys)
        ) # (batch_size, noise_embed_dim)

        # QECCodeEncoder input preprocessing (using its static method from input_encoders.py)
        code_type_ids_tensor, qec_numerical_props_tensor = \
            QECCodeEncoder._preprocess_raw_qec_props(inputs['qec_code_props'], self.qec_code_encoder.code_types)
        qec_embed = self.qec_code_encoder(code_type_ids_tensor, qec_numerical_props_tensor) # (batch_size, qec_embed_dim)

        # CircuitEncoder input preprocessing (using the helper function defined in this file)
        circuit_embed = self.circuit_encoder(
            _preprocess_raw_circuit_ops(inputs['circuit_ops'], self.circuit_encoder.max_gates, self.circuit_encoder.gate_types)
        ) # (batch_size, circuit_embed_dim) 

        # Concatenate embeddings for a single "token" per batch item for Transformer input
        combined_features_before_gnn = tf.concat([noise_embed, qec_embed, circuit_embed], axis=-1)

        graph_embed = self.gnn_placeholder(combined_features_before_gnn) 

        final_combined_features = tf.concat([combined_features_before_gnn, graph_embed], axis=-1)

        transformer_input = tf.expand_dims(final_combined_features, axis=1) 
        
        transformer_output = self.transformer_block(transformer_input, training=training)
        
        fused_features = layers.Flatten()(transformer_output) 
        fused_features = self.fusion_layer(fused_features)

        mitigation_strategy = self.mitigation_output(fused_features)
        decoder_choice = self.decoder_output(fused_features)
        compiler_action = self.compiler_output(fused_features)

        return {
            'mitigation_strategy': mitigation_strategy,
            'decoder_choice': decoder_choice,
            'compiler_action': compiler_action
        }