# src/qera_core/ues_model/transformer_gnn_core.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Any, List, Union

from src.qera_core.ues_model.input_encoders import NoiseEncoder, QECCodeEncoder, CircuitEncoder

# --- DEBUG PRINT: Confirm which CircuitEncoder is imported ---
print(f"--- DEBUG: CircuitEncoder is imported from: {CircuitEncoder.__module__}") 

# --- Placeholder for a basic GNN layer (MOVED TO THE TOP) ---
class SimpleGraphConv(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(output_dim, activation='relu')

    def call(self, inputs):
        # Inputs to this simple GNN will be a (batch_size, feature_dim) tensor
        # For Phase 1, it's just a dense layer processing node features.
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
        # inputs shape: (batch_size, sequence_length, embed_dim)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


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
        
        # Instantiate SimpleGraphConv AFTER its definition
        self.gnn_placeholder = SimpleGraphConv(config['graph_embed_dim']) 
        
        # Calculate total combined embedding dimension for the Transformer's input token
        # This needs to be consistent with the actual output dimensions of the encoders + GNN
        # NoiseEncoder output_dim = config['noise_embed_dim']
        # QECCodeEncoder output_dim = config['qec_embed_dim']
        # CircuitEncoder output_dim = config['gate_embed_dim'] * 2 (because GRU output_dim is 2x embed_dim in config)
        # SimpleGraphConv output_dim = config['graph_embed_dim']
        
        total_combined_embed_dim = config['noise_embed_dim'] + \
                                   config['qec_embed_dim'] + \
                                   (config['gate_embed_dim'] * 2) + \
                                   config['graph_embed_dim'] # This sum MUST match what concat produces

        self.transformer_block = TransformerBlock(
            embed_dim=total_combined_embed_dim, # Set embed_dim to be the total combined feature size
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
        
        # Encoders handle batching by themselves if they get single dict/list or list of dict/list
        noise_embed = self.noise_encoder(inputs['noise_data']) # (batch_size, noise_embed_dim)
        qec_embed = self.qec_code_encoder(inputs['qec_code_props']) # (batch_size, qec_embed_dim)
        circuit_embed = self.circuit_encoder(inputs['circuit_ops']) # (batch_size, circuit_embed_dim) (from GRU)

        # Concatenate embeddings for the 'input token' to the Transformer
        # Shape: (batch_size, sum_of_encoder_output_dims)
        combined_features_before_gnn = tf.concat([noise_embed, qec_embed, circuit_embed], axis=-1)

        # GNN placeholder processes this combined feature vector
        graph_embed = self.gnn_placeholder(combined_features_before_gnn) # (batch_size, graph_embed_dim)

        # Final concatenation for the single Transformer token
        final_combined_features = tf.concat([combined_features_before_gnn, graph_embed], axis=-1)

        # Transformer input needs to be 3D: (batch_size, sequence_length, embed_dim)
        # We treat the entire combined feature vector as a single "token" (sequence_length = 1).
        transformer_input = tf.expand_dims(final_combined_features, axis=1) # (batch_size, 1, total_combined_embed_dim)
        
        transformer_output = self.transformer_block(transformer_input, training=training)
        
        # Flatten the transformer output for the fusion layer
        fused_features = layers.Flatten()(transformer_output) # Flattens from (batch, 1, embed_dim) to (batch, embed_dim)
        fused_features = self.fusion_layer(fused_features)

        # Output QEC strategy components
        mitigation_strategy = self.mitigation_output(fused_features)
        decoder_choice = self.decoder_output(fused_features)
        compiler_action = self.compiler_output(fused_features)

        return {
            'mitigation_strategy': mitigation_strategy,
            'decoder_choice': decoder_choice,
            'compiler_action': compiler_action
        }