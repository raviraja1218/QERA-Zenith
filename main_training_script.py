import sys
import os
import tensorflow as tf
import time # Import time for sleep
import datetime # Import datetime for logging_utils

# NEW IMPORTS for Phase 2: Qiskit NoiseModel
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
# You might also need IBMQ here for real device data in a later step of Phase 2,
# but for now, we'll create a synthetic NoiseModel.
# from qiskit import IBMQ # Will be used in Phase 2, Section A, Step 2

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import core modules from src/qera_core
from src.qera_core.rl_agent.environment_wrapper import QECEnvironment
from src.qera_core.rl_agent.agent_logic import UES_RL_Agent
from src.qera_core.ues_model.transformer_gnn_core import UESModel
from src.qera_core.utils.logging_utils import setup_tensorboard_logger, log_episode_metrics

# --- Configuration ---
NUM_PHYSICAL_QUBITS = 3 # For 3-qubit repetition code example
# NOISE_CONFIG is now largely replaced by qiskit_noise_model for simulator,
# but we'll keep it for UES input encoding (which expects a dict of general noise parameters)
NOISE_CONFIG = {
    'depolarizing_gate_1q': 0.005, # Example 1-qubit gate depolarizing error
    'depolarizing_gate_2q': 0.01,  # Example 2-qubit gate depolarizing error
    'readout_error': 0.01,        # Example readout error
    't1_avg': 100e-6,             # Average T1 coherence time
    't2_avg': 80e-6,              # Average T2 coherence time
    'crosstalk_strength_avg': 0.001 # Average crosstalk strength
}
NUM_EPISODES = 100 # Number of training episodes
MAX_STEPS_PER_EPISODE = 5 # As defined in environment_wrapper
LOG_DIR = 'logs/qera_training' # Directory for TensorBoard logs

# UES Model Configuration (simplified for Phase 1, but with Phase 2 sizing)
UES_CONFIG = {
    'noise_embed_dim': 16, # Output dimension of NoiseEncoder
    'qec_embed_dim': 8,    # Output dimension of QECCodeEncoder
    'max_circuit_gates': 5, # Max operations in simplified circuit for CircuitEncoder
    'gate_embed_dim': 8,    # Embedding dimension for individual gates in CircuitEncoder
    'transformer_embed_dim': 64, # Needs to match total combined input dims for first token
    'transformer_num_heads': 2,
    'transformer_ff_dim': 128,
    'graph_embed_dim': 16, # Output dimension of SimpleGraphConv
    'fusion_dim': 128,     # Output dimension after fusion layer
    'num_mitigation_actions': 4, # 'none', 'zne', 'pec', 'dynamical_decoupling'
    'num_decoder_choices': 3,    # 'basic_mwpm', 'neural_decoder_placeholder', 'lookup_table'
    'num_compiler_actions': 3    # 'default_mapping', 'swap_router', 'optimization_pass_A'
}

# --- Main Training Loop ---
if __name__ == "__main__":
    print("Starting main_training_script.py execution...")

    # Initialize summary_writer early and robustly
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
        # --- Phase 2: Create a Qiskit Aer NoiseModel ---
        # This will be passed to LogicalQubitSimulator via QECEnvironment
        qiskit_aer_noise_model = NoiseModel()
        # Add a 1-qubit depolarizing error to all 1-qubit gates
        qiskit_aer_noise_model.add_all_qubit_quantum_error(
            depolarizing_error(NOISE_CONFIG['depolarizing_gate_1q'], 1), ['h', 'x', 'y', 'z']
        )
        # Add a 2-qubit depolarizing error to all 2-qubit gates
        qiskit_aer_noise_model.add_all_qubit_quantum_error(
            depolarizing_error(NOISE_CONFIG['depolarizing_gate_2q'], 2), ['cx']
        )
        # Add a basic readout error
        readout_err = ReadoutError([[1 - NOISE_CONFIG['readout_error'], NOISE_CONFIG['readout_error']], 
                                    [NOISE_CONFIG['readout_error'], 1 - NOISE_CONFIG['readout_error']]])
        qiskit_aer_noise_model.add_all_qubit_readout_error(readout_err)


        # Initialize Environment and Agent
        # IMPORTANT: Pass qiskit_aer_noise_model to QECEnvironment
        env = QECEnvironment(num_physical_qubits=NUM_PHYSICAL_QUBITS, qiskit_noise_model=qiskit_aer_noise_model)
        
        ues_model = UESModel(UES_CONFIG)
        agent = UES_RL_Agent(ues_model, learning_rate=1e-3)

        # Build UES model by passing dummy input once
        # dummy_noise_data_for_encoder should match NoiseEncoder's expected_noise_keys
        dummy_noise_data_for_encoder = {
            'gate_error_1q': NOISE_CONFIG['depolarizing_gate_1q'],
            'gate_error_2q': NOISE_CONFIG['depolarizing_gate_2q'],
            'readout_error': NOISE_CONFIG['readout_error'],
            't1_avg': NOISE_CONFIG['t1_avg'],
            't2_avg': NOISE_CONFIG['t2_avg'],
            'crosstalk_strength_avg': NOISE_CONFIG['crosstalk_strength_avg']
        }
        
        dummy_qec_props = {'code_type':'repetition', 'distance':3, 'num_physical_qubits':3, 'num_logical_qubits':1}
        dummy_circuit_ops = [{'type':'gate', 'name':'h', 'qubits':[0]}] # Must match expected format for CircuitEncoder
        
        dummy_inputs = {
            'noise_data': dummy_noise_data_for_encoder, 
            'qec_code_props': dummy_qec_props, 
            'circuit_ops': dummy_circuit_ops
        }
        # Pass dummy input to build model's graph. This call triggers the UESModel.call method.
        _ = ues_model(dummy_inputs) 
        ues_model.summary() # Print model summary

        print(f"Starting QERA Phase 1 RL Training for {NUM_EPISODES} episodes...")

        for episode in range(NUM_EPISODES):
            observation = env.reset()
            done = False
            episode_rewards = []
            experiences = []

            while not done:
                action = agent.choose_action(observation)
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
            
            time.sleep(0.1) # Small sleep to ensure logs are written and visible

        print("✅ TRAINING COMPLETED — QERA-Zenith Phase 1 end-to-end pipeline ran successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    finally: # This block always executes, even if an error occurs
        if summary_writer: # Only flush/close if it was successfully initialized
            summary_writer.flush()
            summary_writer.close()
        print(f"TensorBoard logs should be available at: tensorboard --logdir {LOG_DIR}")
        print("Script finished!")
