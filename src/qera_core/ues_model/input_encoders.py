# src/qera_core/ues_model/input_encoders.py
import tensorflow as tf  # <--- CRITICAL FIX: ADD THIS LINE
import numpy as np
from typing import Dict, Any, List, Union

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
    """Encodes quantum circuit structure (simplified for Phase 1)."""
    def __init__(self, max_gates: int = 20, gate_embedding_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_gates = max_gates
        # Updated input_dim based on your gate_types dict size (6 values)
        self.gate_types = {'h':0, 'x':1, 'cx':2, 'measure':3, 'identity':4, 'other':5} 
        self.gate_embedding = tf.keras.layers.Embedding(input_dim=len(self.gate_types), output_dim=gate_embedding_dim)
        
        self.gru_layer = tf.keras.layers.GRU(gate_embedding_dim * 2) # GRU output will be 2 * gate_embedding_dim

    def call(self, circuit_ops_batch: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        """
        :param circuit_ops_batch: A single list of ops (for one circuit) or a list of lists of ops (batch of circuits).
        :return: Encoded tensor (batch_size, GRU_output_dim).
        """
        print(f"\n--- CircuitEncoder.call received raw input: {circuit_ops_batch}")
        print(f"--- CircuitEncoder.call input type: {type(circuit_ops_batch)}")
        if isinstance(circuit_ops_batch, list) and len(circuit_ops_batch) > 0 and isinstance(circuit_ops_batch[0], dict):
            # If it's a single circuit (list of dicts), convert to batch of 1
            circuit_ops_batch = [circuit_ops_batch]
            print(f"--- CircuitEncoder: Converted single circuit to batch: {circuit_ops_batch}")
        elif not isinstance(circuit_ops_batch, list) or (circuit_ops_batch and not isinstance(circuit_ops_batch[0], list)):
            # This should handle cases where the input is a single dict (incorrect, but for robustness)
            # Or other non-list-of-list-of-dict cases.
            # For our dummy input, it should be `[[{'type': 'gate', 'name': 'h', 'qubits': [0]}]]`
            print(f"--- CircuitEncoder: Adjusting unexpected input format: {type(circuit_ops_batch)}")
            if isinstance(circuit_ops_batch, dict): # If somehow got just the dict, wrap it twice.
                circuit_ops_batch = [[circuit_ops_batch]]
            elif isinstance(circuit_ops_batch, list) and len(circuit_ops_batch) > 0 and not isinstance(circuit_ops_batch[0], list): # list of dicts, but expected list of list of dicts
                circuit_ops_batch = [circuit_ops_batch]
            
        batched_gate_ids = []
        for circuit_ops in circuit_ops_batch: # Iterate through each circuit in the batch
            gate_ids_single_circuit = []
            for op in circuit_ops:
                gate_name = op.get('name', 'other')
                # This print should show the string 'h' for the dummy input
                print(f"--- CircuitEncoder: Processing operation name: '{gate_name}'") 
                gate_id = self.gate_types.get(gate_name, self.gate_types['other'])
                gate_ids_single_circuit.append(gate_id)
            
            # Pad/truncate sequence for this single circuit
            padded_gate_ids = gate_ids_single_circuit + [self.gate_types['other']] * (self.max_gates - len(gate_ids_single_circuit))
            padded_gate_ids = padded_gate_ids[:self.max_gates]
            
            batched_gate_ids.append(padded_gate_ids)
        
        # This print should show the numerical ID, not 'h'
        print(f"--- CircuitEncoder: batched_gate_ids before tf.constant: {batched_gate_ids}") 
        input_tensor_for_embedding = tf.constant(batched_gate_ids, dtype=tf.int32)
        print(f"--- CircuitEncoder: input_tensor_for_embedding shape: {input_tensor_for_embedding.shape}, dtype: {input_tensor_for_embedding.dtype}")
        
        embedded_sequences = self.gate_embedding(input_tensor_for_embedding)
        print(f"--- CircuitEncoder: embedded_sequences shape: {embedded_sequences.shape}")
        
        return self.gru_layer(embedded_sequences)