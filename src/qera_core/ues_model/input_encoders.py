# src/qera_core/ues_model/input_encoders.py
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Union

print("--- DEBUG: input_encoders.py is loaded. (Unique ID: Alpha)")

class NoiseEncoder(tf.keras.layers.Layer):
    """Encodes hardware noise parameters into a fixed-size vector."""
    def __init__(self, output_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_layer = tf.keras.layers.Dense(output_dim, activation='relu')
        self.expected_noise_keys = [
            'gate_error_1q', 'gate_error_2q', 'readout_error', 
            't1_avg', 't2_avg', 'crosstalk_strength_avg' 
        ]

    def call(self, noise_data_batch: Union[List[Dict[str, Any]], Dict[str, Any]]):
        """
        :param noise_data_batch: A single dict or a list of dicts.
        :return: Encoded tensor (batch_size, output_dim).
        """
        if isinstance(noise_data_batch, dict): # Single instance, add batch dim
            noise_data_batch = [noise_data_batch]

        input_vectors = []
        for noise_data in noise_data_batch:
            input_vector_single = []
            for key in self.expected_noise_keys:
                input_vector_single.append(noise_data.get(key, 0.0 if 'error' in key else 1e-6))
            input_vectors.append(input_vector_single)
        
        input_tensor = tf.convert_to_tensor(input_vectors, dtype=tf.float32)
        return self.dense_layer(input_tensor)

class QECCodeEncoder(tf.keras.layers.Layer):
    """Encodes QEC code properties into a fixed-size vector."""
    def __init__(self, output_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_layer = tf.keras.layers.Dense(output_dim, activation='relu')
        self.code_type_embedding = tf.keras.layers.Embedding(input_dim=5, output_dim=4)
        self.code_types = {'repetition': 0, 'surface': 1, 'ldpc': 2, 'cat': 3, 'other': 4}

    def call(self, code_properties_batch: Union[List[Dict[str, Any]], Dict[str, Any]]):
        """
        :param code_properties_batch: A single dict or a list of dicts.
        :return: Encoded tensor (batch_size, output_dim).
        """
        if isinstance(code_properties_batch, dict):
            code_properties_batch = [code_properties_batch]

        code_type_ids = [self.code_types.get(props.get('code_type', 'other'), 4) for props in code_properties_batch]
        code_type_emb = self.code_type_embedding(tf.constant(code_type_ids)) # Output: (batch_size, 4)

        numerical_props_list = []
        for props in code_properties_batch:
            numerical_props_list.append([
                props.get('distance', 3),
                props.get('num_physical_qubits', 3),
                props.get('num_logical_qubits', 1)
            ])
        numerical_props_tensor = tf.convert_to_tensor(numerical_props_list, dtype=tf.float32) # Output: (batch_size, 3)

        combined_input = tf.concat([code_type_emb, numerical_props_tensor], axis=-1)
        return self.dense_layer(combined_input)

class CircuitEncoder(tf.keras.layers.Layer):
    """
    Encodes quantum circuit structure.
    For Keras 3 compatibility, its `call` method now expects a tensor of gate IDs directly.
    The parsing from `List[Dict[str,Any]]` to `tf.Tensor` is handled by `_preprocess_raw_circuit_ops`.
    """
    def __init__(self, max_gates: int = 20, gate_embedding_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_gates = max_gates
        self.gate_types = {'h':0, 'x':1, 'cx':2, 'measure':3, 'identity':4, 'other':5} 
        self.gate_embedding = tf.keras.layers.Embedding(input_dim=len(self.gate_types), output_dim=gate_embedding_dim)
        
        # GRU output will be (batch_size, gate_embedding_dim * 2)
        # return_sequences=False makes GRU return the last output, not a sequence of outputs
        self.gru_layer = tf.keras.layers.GRU(gate_embedding_dim * 2, return_sequences=False) 

    def call(self, input_tensor_for_embedding: tf.Tensor): # <-- CHANGED: Expects tf.Tensor directly
        """
        :param input_tensor_for_embedding: A tf.Tensor of shape (batch_size, max_gates) with dtype tf.int32,
                                           containing integer IDs of gates.
        :return: Encoded tensor (batch_size, GRU_output_dim).
        """
        # No more complex parsing here, as input is already a tensor of IDs.
        # This will contain the debug prints from before the last change, but no new ones.

        if input_tensor_for_embedding.dtype != tf.int32:
            raise TypeError(f"CircuitEncoder expects input_tensor_for_embedding to be tf.int32, but got {input_tensor_for_embedding.dtype}")
        if len(input_tensor_for_embedding.shape) != 2:
            raise ValueError(f"CircuitEncoder expects 2D input tensor (batch_size, max_gates), but got shape {input_tensor_for_embedding.shape}")

        # Pass the tensor directly to the embedding layer
        embedded_sequences = self.gate_embedding(input_tensor_for_embedding) # (batch_size, max_gates, embed_dim)
        
        # Pass the embedded sequences to the GRU layer
        return self.gru_layer(embedded_sequences) # (batch_size, GRU_output_dim)

    @staticmethod
    def _preprocess_raw_circuit_ops(
        circuit_ops_batch: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]], 
        max_gates: int, 
        gate_types: Dict[str, int]
    ) -> tf.Tensor:
        """
        Helper method to preprocess raw circuit_ops (list of dicts) into a padded tensor of gate IDs.
        This method will be called BEFORE passing to CircuitEncoder.call.
        :param circuit_ops_batch: A single list of ops (for one circuit) or a list of lists of ops (batch of circuits).
        :param max_gates: Max sequence length for padding.
        :param gate_types: Dictionary mapping gate names to integer IDs.
        :return: tf.Tensor of shape (batch_size, max_gates) with dtype tf.int32.
        """
        # Ensure input is always a list of circuits (even if batch size 1)
        if not isinstance(circuit_ops_batch, list) or (circuit_ops_batch and not isinstance(circuit_ops_batch[0], list)):
            circuit_ops_batch = [circuit_ops_batch]

        batched_gate_ids = []
        for circuit_ops in circuit_ops_batch: # Iterate through each circuit in the batch
            gate_ids_single_circuit = []
            for op in circuit_ops:
                gate_name = op.get('name', 'other')
                gate_id = gate_types.get(gate_name, gate_types['other'])
                gate_ids_single_circuit.append(gate_id)
            
            # Pad/truncate sequence for this single circuit
            padded_gate_ids = gate_ids_single_circuit + [gate_types['other']] * (max_gates - len(gate_ids_single_circuit))
            padded_gate_ids = padded_gate_ids[:max_gates]
            
            batched_gate_ids.append(padded_gate_ids)
        
        # Convert the list of padded ID lists into a single TensorFlow tensor (batch_size, max_gates)
        return tf.constant(batched_gate_ids, dtype=tf.int32)