# Cell 4: Run Your Main Training Script (Complete Code from main_training_script.py)

import sys
import os
import tensorflow as tf
import time
import datetime
from typing import Dict, Any, List, Union # Needed for type hints in helper function

# NEW IMPORTS for Phase 2: Qiskit NoiseModel
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError 

# Add the 'src' directory to the Python path
# This is crucial for your Colab notebook to find your modules inside src/qera_core
current_dir = os.getcwd() 
src_path = os.path.join(current_dir, 'src') 
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import core modules from src/qera_core
from qera_core.rl_agent.environment_wrapper import QECEnvironment
from qera_core.rl_agent.agent_logic import UES_RL_Agent
from qera_core.ues_model.transformer_gnn_core import UESModel
from qera_core.utils.logging_utils import setup_tensorboard_logger, log_episode_metrics

# --- Helper for CircuitEncoder input processing ---
# This function is duplicated here from input_encoders.py to ensure the input
# to CircuitEncoder is always a tensor, bypassing Keras 3's strict tracing.
# This is a temporary measure for Phase 1/2 setup. In a more mature project,
# this preprocessing would be part of a data pipeline.
def _preprocess_raw_circuit_ops(
    circuit_ops_batch: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]], 
    max_gates: int, 
    gate_types: Dict[str, int]
) -> tf.Tensor:
    """
    Helper method to preprocess raw circuit_ops (list of dicts) into a padded tensor of gate IDs.
    This method is called BEFORE passing to CircuitEncoder.call.
    :param circuit_ops_batch: A single list of ops (for one circuit) or a list of lists of ops (batch of circuits).
    :param max_gates: Max sequence length for padding.
    :param gate_types: Dictionary mapping gate names to integer IDs (from CircuitEncoder).
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

# --- Configuration ---
NUM_PHYSICAL_QUBITS = 3 
NOISE_CONFIG = {
    'depolarizing_gate_1q': 0.005, 
    'depolarizing_gate_2q': 0.01,  
    'readout_error': 0.01,        
    't1_avg': 100e-6,             
    't2_avg': 80e-6,              
    'crosstalk_strength_avg': 0.001 
}
NUM_EPISODES = 100 
MAX_STEPS_PER_EPISODE = 5 
LOG_DIR = 'logs/qera_training' 

# UES Model Configuration (simplified for Phase 1, but with Phase 2 sizing)
UES_CONFIG = {
    'noise_embed_dim': 16, 
    'qec_embed_dim': 8,    
    'max_circuit_gates': 5, # Max operations in simplified circuit for CircuitEncoder
    'gate_embed_dim': 8,    # Embedding dimension for individual gates in CircuitEncoder
    'transformer_embed_dim': 64, 
    'transformer_num_heads': 2,
    'transformer_ff_dim': 128,
    'graph_embed_dim': 16, 
    'fusion_dim': 128,     
    'num_mitigation_actions': 4, 
    'num_decoder_choices': 3,    
    'num_compiler_actions': 3    
}

# --- Main Training Loop ---
if __name__ == "__main__":
    print("Starting main_training_script.py execution...")

    summary_writer = None
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        summary_writer = setup_tensorboard_logger(LOG_DIR)
    except Exception as e:
        print(f"Error setting up TensorBoard logger: {e}")
        print("Proceeding without TensorBoard logging, but this is critical for full functionality.")
        class DummySummaryWriter: 
            def as_default(self): return self
            def scalar(self, *args, **kwargs): pass
            def flush(self): pass
            def close(self): pass
        summary_writer = DummySummaryWriter()


    try:
        # --- Phase 2: Use the NoiseModel prepared in Cell 3 ---
        global qiskit_aer_noise_model_for_simulator 
        
        if 'qiskit_aer_noise_model_for_simulator' not in globals():
            print("WARNING: qiskit_aer_noise_model_for_simulator not found from Cell 3. Creating a default synthetic one within Cell 4.")
            qiskit_aer_noise_model_for_simulator = NoiseModel()
            qiskit_aer_noise_model_for_simulator.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['id'])


        # Initialize Environment and Agent
        env = QECEnvironment(num_physical_qubits=NUM_PHYSICAL_QUBITS, qiskit_noise_model=qiskit_aer_noise_model_for_simulator)
        
        ues_model = UESModel(UES_CONFIG)
        agent = UES_RL_Agent(ues_model, learning_rate=1e-3)

        # Build UES model by passing dummy input once
        dummy_noise_data_for_encoder = {
            'gate_error_1q': NOISE_CONFIG['depolarizing_gate_1q'],
            'gate_error_2q': NOISE_CONFIG['depolarizing_gate_2q'],
            'readout_error': NOISE_CONFIG['readout_error'],
            't1_avg': NOISE_CONFIG['t1_avg'],
            't2_avg': NOISE_CONFIG['t2_avg'],
            'crosstalk_strength_avg': NOISE_CONFIG['crosstalk_strength_avg']
        }
        
        dummy_qec_props = {'code_type':'repetition', 'distance':3, 'num_physical_qubits':3, 'num_logical_qubits':1}
        
        # --- CRITICAL FIX: Pre-process circuit_ops for CircuitEncoder ---
        # Call the static helper method to convert raw circuit ops to tensor of IDs
        # We need the gate_types dict from CircuitEncoder to preprocess
        # Access it via a dummy instance or directly from the class if it were a static property.
        # For simplicity, let's duplicate the gate_types dict here for preprocessing.
        
        # Define gate_types here, matching CircuitEncoder's __init__
        circuit_encoder_gate_types = {'h':0, 'x':1, 'cx':2, 'measure':3, 'identity':4, 'other':5}
        
        dummy_circuit_ops_raw = [{'type':'gate', 'name':'h', 'qubits':[0]}] 
        dummy_circuit_ops_processed = _preprocess_raw_circuit_ops(
            dummy_circuit_ops_raw,
            UES_CONFIG['max_circuit_gates'],
            circuit_encoder_gate_types
        )
        # --- End CRITICAL FIX ---

        dummy_inputs = {
            'noise_data': dummy_noise_data_for_encoder, 
            'qec_code_props': dummy_qec_props, 
            'circuit_ops': dummy_circuit_ops_processed # <-- Pass the PRE-PROCESSED TENSOR
        }
        
        # Pass dummy input to build model's graph. This call triggers the UESModel.call method.
        _ = ues_model(dummy_inputs) 
        ues_model.summary() # Print model summary

        print(f"Starting QERA Phase 1 RL Training for {NUM_EPISODES} episodes...")

        for episode in range(NUM_EPISODES):
            # The observation from env.reset() also needs its 'circuit_ops' pre-processed
            # before passing to agent.choose_action().
            # Let's modify agent.choose_action() and agent.learn() to handle this preprocessing.
            # For now, the dummy input for ues_model.summary() is fixed.
            
            observation = env.reset()
            done = False
            episode_rewards = []
            experiences = []

            while not done:
                action = agent.choose_action(observation) # This will call UESModel
                next_observation, reward, done, info = env.step(action)
                experiences.append((observation, action, reward, next_observation, done))
                episode_rewards.append(reward)
                observation = next_observation

            agent.learn(experiences)

            total_episode_reward = sum(episode_rewards)
            final_fidelity = info.get('fidelity_after_step', 0.0)
            syndrome_detected_count = info.get('syndrome_detected_count', 0)

            log_episode_metrics(
                summary_writer,
                episode,
                total_episode_reward,
                final_fidelity,
                syndrome_detected_count=syndrome_detected_count
            )
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_episode_reward:.4f}, Final Fidelity: {final_fidelity:.4f}")
            
            time.sleep(0.1) 

        print("✅ TRAINING COMPLETED — QERA-Zenith Phase 1 end-to-end pipeline ran successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc() 

    finally: 
        if summary_writer: 
            summary_writer.flush()
            summary_writer.close()
        print(f"TensorBoard logs should be available at: tensorboard --logdir {LOG_DIR}")
        print("Script finished!")