# src/qera_core/rl_agent/agent_logic.py
import tensorflow as tf
# REMOVED: from tensorflow.keras.optimizers import legacy # This line is removed as it's no longer supported in Keras 3 (TF 2.16+)

from src.qera_core.ues_model.transformer_gnn_core import UESModel 

from typing import Dict, Any, List, Tuple # Needed for type hints in learn method
import numpy as np # Correctly imported now


class UES_RL_Agent: 
    """
    A basic Reinforcement Learning agent for the UESModel (policy network).
    Implements a simple policy gradient approach for Phase 1.
    """
    def __init__(self, ues_model: UESModel, learning_rate: float = 1e-3):
        self.ues_model = ues_model
        
        # --- CRITICAL FIX HERE: Use tf.keras.optimizers.Adam directly ---
        # With TensorFlow 2.16+ (which you have), Keras 3 is used.
        # tf.keras.optimizers.legacy is removed. Use tf.keras.optimizers.Adam directly.
        # TensorFlow will handle CPU/GPU optimization internally for Apple Silicon.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
        
        # Use CategoricalCrossentropy as your UESModel outputs softmax probabilities (from_logits=False)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 

    def choose_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses the UES model to predict the best QEC strategy (action) given the current observation.
        :param observation: Dictionary containing noise data, QEC code props, circuit ops.
        :return: Dictionary of chosen QEC strategy components (integer indices).
        """
        # Get raw probabilities from UESModel for a single observation.
        # The encoders in ues_model.input_encoders.py are designed to handle
        # single dictionary inputs and internally add a batch dimension of 1.
        predictions = self.ues_model({
            'noise_data': observation['noise_data'],
            'qec_code_props': observation['qec_code_props'],
            'circuit_ops': observation['circuit_ops']
        }, training=False) # Important: set training=False for inference mode


        # Convert softmax outputs (which have a batch dimension of 1) to discrete actions (integer indices).
        # We use [0] to remove the batch dimension from the output tensors.
        action = {
            'mitigation_strategy': tf.argmax(predictions['mitigation_strategy'][0]).numpy(),
            'decoder_choice': tf.argmax(predictions['decoder_choice'][0]).numpy(),
            'compiler_action': tf.argmax(predictions['compiler_action'][0]).numpy()
        }
        return action

    def learn(self, experiences: List[Tuple[Dict, Dict, float, Dict, bool]]):
        """
        Updates the UES model's weights based on collected experiences.
        For Phase 1, implements a very basic policy gradient update (REINFORCE-like) using
        cross-entropy loss as a proxy. This is simplified and will be refined in Phase 2.
        :param experiences: List of (observation, action, reward, next_observation, done) tuples.
        """
        if not experiences:
            return # Nothing to learn from

        # Prepare batched inputs and targets from experiences
        obs_batch_noise = [exp[0]['noise_data'] for exp in experiences]
        obs_batch_qec = [exp[0]['qec_code_props'] for exp in experiences]
        obs_batch_circuit = [exp[0]['circuit_ops'] for exp in experiences]
        
        # Prepare action batch (integer indices) and reward batch
        action_batch = {
            'mitigation_strategy': [exp[1]['mitigation_strategy'] for exp in experiences],
            'decoder_choice': [exp[1]['decoder_choice'] for exp in experiences],
            'compiler_action': [exp[1]['compiler_action'] for exp in experiences]
        }
        reward_batch = tf.constant([exp[2] for exp in experiences], dtype=tf.float32)

        # Apply basic reward normalization for stability (crucial in RL)
        if tf.math.reduce_std(reward_batch) > 1e-6: # Avoid division by zero
            reward_batch = (reward_batch - tf.reduce_mean(reward_batch)) / tf.math.reduce_std(reward_batch)
        else:
            reward_batch = tf.zeros_like(reward_batch) # If all rewards are same, set to zero

        with tf.GradientTape() as tape:
            # Get predicted probabilities from UESModel for the batch of observations
            predicted_actions_probs = self.ues_model({
                'noise_data': obs_batch_noise,
                'qec_code_props': obs_batch_qec,
                'circuit_ops': obs_batch_circuit
            }, training=True) # Set training=True for learning

            # Convert discrete integer actions from `action_batch` to one-hot encoded targets
            num_mitigation = predicted_actions_probs['mitigation_strategy'].shape[-1]
            num_decoder = predicted_actions_probs['decoder_choice'].shape[-1]
            num_compiler = predicted_actions_probs['compiler_action'].shape[-1]

            target_mitigation = tf.one_hot(action_batch['mitigation_strategy'], depth=num_mitigation)
            target_decoder = tf.one_hot(action_batch['decoder_choice'], depth=num_decoder)
            target_compiler = tf.one_hot(action_batch['compiler_action'], depth=num_compiler)

            # Calculate individual component losses
            mitigation_loss = self.loss_fn(target_mitigation, predicted_actions_probs['mitigation_strategy'])
            decoder_loss = self.loss_fn(target_decoder, predicted_actions_probs['decoder_choice'])
            compiler_loss = self.loss_fn(target_compiler, predicted_actions_probs['compiler_action'])
            
            # Simplified weighted loss:
            weighted_mitigation_loss = mitigation_loss * reward_batch
            weighted_decoder_loss = decoder_loss * reward_batch
            weighted_compiler_loss = compiler_loss * reward_batch
            
            total_weighted_loss = tf.reduce_mean(weighted_mitigation_loss + weighted_decoder_loss + weighted_compiler_loss)

        # Apply gradients based on the total weighted loss
        grads = tape.gradient(total_weighted_loss, self.ues_model.trainable_variables)
        # Clip gradients to prevent exploding gradients (common in RL)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.ues_model.trainable_variables))