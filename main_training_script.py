# Cell 4: Run Your Main Training Script (Complete Code from main_training_script.py)

import sys
import os
import tensorflow as tf
import time
import datetime
from typing import Dict, Any, List, Union 

# NEW IMPORTS for Phase 2: Qiskit NoiseModel
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError 

# Add the 'src' directory to the Python path
# This is crucial for your Colab notebook to find your modules inside src/qera_core
current_dir = os.getcwd() 
src_path = os.path.join(current_dir, 'src') 
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import core modules from src/qera_core
# Note: input_encoders is NOT imported here directly anymore, as its static methods are called via the Encoder classes
from qera_core.rl_agent.environment_wrapper import QECEnvironment
from qera_core.rl_agent.agent_logic import UES_RL_Agent
from qera_core.ues_model.transformer_gnn_core import UESModel
# REMOVED: from qera_core.ues_model.input_encoders import NoiseEncoder, QECCodeEncoder, CircuitEncoder 
# The static preprocess methods are now called via the UESModel's encoders (e.g., NoiseEncoder._preprocess_raw_noise_data)
# or are part of transformer_gnn_core.py itself (_preprocess_raw_circuit_ops).
from qera_core.utils.logging_utils import setup_tensorboard_logger, log_episode_metrics

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
    'max_circuit_gates': 5, 
    'gate_embed_dim': 8,    
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

        # Build UES model by passing dummy input once (in RAW format)
        # UESModel.call() will now handle preprocessing these raw inputs internally.
        
        dummy_noise_data_for_encoder_raw = {
            'gate_error_1q': NOISE_CONFIG['depolarizing_gate_1q'],
            'gate_error_2q': NOISE_CONFIG['depolarizing_gate_2q'],
            'readout_error': NOISE_CONFIG['readout_error'],
            't1_avg': NOISE_CONFIG['t1_avg'],
            't2_avg': NOISE_CONFIG['t2_avg'],
            'crosstalk_strength_avg': NOISE_CONFIG['crosstalk_strength_avg']
        }
        
        dummy_qec_props_raw = {'code_type':'repetition', 'distance':3, 'num_physical_qubits':3, 'num_logical_qubits':1}
        dummy_circuit_ops_raw = [{'type':'gate', 'name':'h', 'qubits':[0]}] 
        
        dummy_inputs_raw_format = {
            'noise_data': dummy_noise_data_for_encoder_raw, 
            'qec_code_props': dummy_qec_props_raw, 
            'circuit_ops': dummy_circuit_ops_raw 
        }
        
        # Pass dummy input in RAW format. UESModel.call will preprocess.
        _ = ues_model(dummy_inputs_raw_format) 
        ues_model.summary() # Print model summary

        print(f"Starting QERA Phase 1 RL Training for {NUM_EPISODES} episodes...")

        for episode in range(NUM_EPISODES):
            # All observations coming from env.reset() and env.step() are in RAW dictionary/list format.
            # The UESModel.call method (called by agent.choose_action and agent.learn)
            # now expects and handles this raw format by internally calling the static preprocessors.
            
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