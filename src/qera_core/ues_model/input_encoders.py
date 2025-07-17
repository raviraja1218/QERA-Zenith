# src/qera_core/ues_model/input_encoders.py
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Union, Tuple # Added Tuple for type hints

print("--- DEBUG: input_encoders.py is loaded. (Unique ID: Alpha)")

class NoiseEncoder(tf.keras.layers.Layer):
    """
    Encodes hardware noise parameters.
    Its `call` method now expects a tensor of numerical features.
    Preprocessing from raw dicts to tensor is handled by `_preprocess_raw_noise_data`.
    """
    def __init__(self, output_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_layer = tf.keras.layers.Dense(output_dim, activation='relu')
        # expected_noise_keys are now used by the static preprocessing method
        self.expected_noise_keys = [ 
            'gate_error_1q', 'gate_error_2q', 'readout_error', 
            't1_avg', 't2_avg', 'crosstalk_strength_avg' 
        ]

    def call(self, input_tensor_for_dense: tf.Tensor): # <-- CHANGED: Expects tf.Tensor directly
        """
        :param input_tensor_for_dense: A tf.Tensor of shape (batch_size, num_features) with dtype tf.float32,
                                        containing pre-processed noise features.
        :return: Encoded tensor (batch_size, output_dim).
        """
        if input_tensor_for_dense.dtype != tf.float32:
            raise TypeError(f"NoiseEncoder expects input to be tf.float32, but got {input_tensor_for_dense.dtype}")
        if len(input_tensor_for_dense.shape) != 2:
            raise ValueError(f"NoiseEncoder expects 2D input tensor (batch_size, num_features), but got shape {input_tensor_for_dense.shape}")
        
        return self.dense_layer(input_tensor_for_dense)

    @staticmethod # <-- NEW STATIC METHOD
    def _preprocess_raw_noise_data(
        noise_data_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
        expected_keys: List[str]
    ) -> tf.Tensor:
        """
        Helper to convert raw noise data dictionaries into a padded TensorFlow tensor.
        This method should be called BEFORE passing to NoiseEncoder.call.
        """
        if isinstance(noise_data_batch, dict):
            noise_data_batch = [noise_data_batch]

        input_vectors = []
        for noise_data in noise_data_batch:
            input_vector_single = []
            for key in expected_keys:
                input_vector_single.append(noise_data.get(key, 0.0 if 'error' in key else 1e-6))
            input_vectors.append(input_vector_single)
        
        return tf.convert_to_tensor(input_vectors, dtype=tf.float32)


class QECCodeEncoder(tf.keras.layers.Layer):
    """
    Encodes QEC code properties.
    Its `call` method now expects pre-processed tensors.
    Preprocessing from raw dicts to tensors is handled by `_preprocess_raw_qec_props`.
    """
    def __init__(self, output_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_layer = tf.keras.layers.Dense(output_dim, activation='relu')
        self.code_type_embedding = tf.keras.layers.Embedding(input_dim=5, output_dim=4)
        # code_types are now used by the static preprocessing method
        self.code_types = {'repetition': 0, 'surface': 1, 'ldpc': 2, 'cat': 3, 'other': 4}

    def call(self, code_type_ids_tensor: tf.Tensor, numerical_props_tensor: tf.Tensor): # <-- CHANGED: Expects Tensors
        """
        :param code_type_ids_tensor: A tf.Tensor of shape (batch_size,) with dtype tf.int32.
        :param numerical_props_tensor: A tf.Tensor of shape (batch_size, num_numerical_features) with dtype tf.float32.
        :return: Encoded tensor (batch_size, output_dim).
        """
        if code_type_ids_tensor.dtype != tf.int32:
            raise TypeError(f"QECCodeEncoder expects code_type_ids_tensor to be tf.int32, but got {code_type_ids_tensor.dtype}")
        if len(code_type_ids_tensor.shape) != 1:
            raise ValueError(f"QECCodeEncoder expects 1D code_type_ids_tensor (batch_size,), but got shape {code_type_ids_tensor.shape}")
        if numerical_props_tensor.dtype != tf.float32:
            raise TypeError(f"QECCodeEncoder expects numerical_props_tensor to be tf.float32, but got {numerical_props_tensor.dtype}")
        if len(numerical_props_tensor.shape) != 2:
            raise ValueError(f"QECCodeEncoder expects 2D numerical_props_tensor (batch_size, num_features), but got shape {numerical_props_tensor.shape}")

        code_type_emb = self.code_type_embedding(code_type_ids_tensor) 
        combined_input = tf.concat([code_type_emb, numerical_props_tensor], axis=-1)
        return self.dense_layer(combined_input)

    @staticmethod # <-- NEW STATIC METHOD
    def _preprocess_raw_qec_props(
        code_properties_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
        code_types_map: Dict[str, int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Helper to convert raw QEC properties dictionaries into separate TensorFlow tensors for embedding and dense layer.
        This method should be called BEFORE passing to QECCodeEncoder.call.
        """
        if isinstance(code_properties_batch, dict):
            code_properties_batch = [code_properties_batch]

        code_type_ids = [code_types_map.get(props.get('code_type', 'other'), 4) for props in code_properties_batch]
        
        numerical_props_list = []
        for props in code_properties_batch:
            numerical_props_list.append([
                props.get('distance', 3),
                props.get('num_physical_qubits', 3),
                props.get('num_logical_qubits', 1)
            ])
        
        return tf.constant(code_type_ids, dtype=tf.int32), tf.convert_to_tensor(numerical_props_list, dtype=tf.float32)


class CircuitEncoder(tf.keras.layers.Layer):
    """
    Encodes quantum circuit structure.
    Its `call` method now expects a tensor of gate IDs directly.
    The raw data preprocessing is handled by `_preprocess_raw_circuit_ops` helper function (defined outside this class).
    """
    def __init__(self, max_gates: int = 20, gate_embedding_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_gates = max_gates
        self.gate_types = {'h':0, 'x':1, 'cx':2, 'measure':3, 'identity':4, 'other':5} 
        self.gate_embedding = tf.keras.layers.Embedding(input_dim=len(self.gate_types), output_dim=gate_embedding_dim)
        
        self.gru_layer = tf.keras.layers.GRU(gate_embedding_dim * 2, return_sequences=False) 

    def call(self, input_tensor_for_embedding: tf.Tensor): # <-- Expects tf.Tensor directly
        """
        :param input_tensor_for_embedding: A tf.Tensor of shape (batch_size, max_gates) with dtype tf.int32,
                                           containing integer IDs of gates.
        :return: Encoded tensor (batch_size, GRU_output_dim).
        """
        if input_tensor_for_embedding.dtype != tf.int32:
            raise TypeError(f"CircuitEncoder expects input_tensor_for_embedding to be tf.int32, but got {input_tensor_for_embedding.dtype}")
        if len(input_tensor_for_embedding.shape) != 2:
            raise ValueError(f"CircuitEncoder expects 2D input tensor (batch_size, max_gates), but got shape {input_tensor_for_embedding.shape}")

        embedded_sequences = self.gate_embedding(input_tensor_for_embedding)
        return self.gru_layer(embedded_sequences)

    # REMOVED @staticmethod _preprocess_raw_circuit_ops from here. It is moved to transformer_gnn_core.py.