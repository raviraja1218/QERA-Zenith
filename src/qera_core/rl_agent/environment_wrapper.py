# src/qera_core/rl_agent/environment_wrapper.py
import numpy as np
from typing import Tuple, Dict, Any, List

# Import core simulation and QEC modules
from src.qera_core.logical_qubit_simulator import LogicalQubitSimulator
from src.qera_core.qec_codes.code_definitions import ThreeQubitRepetitionCode # For Phase 1
from src.qera_core.qec_codes.syndrome_extraction import extract_syndrome_from_state
from src.qera_core.qec_codes.decoders import decode_three_qubit_repetition_code
from qiskit_aer.noise import NoiseModel # <-- NEW IMPORT for Phase 2


class QECEnvironment:
    """
    An OpenAI Gym-like environment for Quantum Error Correction optimization.
    This environment leverages the LogicalQubitSimulator which is now Qiskit-Aer based.
    """
    def __init__(self, num_physical_qubits: int, qiskit_noise_model: NoiseModel = None): # <-- CHANGED: Accepts Qiskit NoiseModel
        self.num_physical_qubits = num_physical_qubits
        self.qiskit_noise_model = qiskit_noise_model # Store the Qiskit NoiseModel

        # Pass the qiskit_noise_model to the LogicalQubitSimulator
        self.simulator = LogicalQubitSimulator(num_physical_qubits, qiskit_noise_model=qiskit_noise_model)
        
        # For Phase 1, use the 3-qubit repetition code as the default QEC code
        self.qec_code = ThreeQubitRepetitionCode() 
        
        # Define the ideal encoded logical state (e.g., |000> for 3-qubit code) for fidelity calculation
        self.ideal_encoded_state = np.zeros(2**num_physical_qubits, dtype=complex)
        self.ideal_encoded_state[0] = 1.0 # |000> state vector

        self.current_step = 0
        self.max_steps_per_episode = 5 # Simulating a short QEC cycle (e.g., 5 rounds of noisy logical operation + QEC)

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment for a new episode.
        Initializes the logical qubit in the simulator.
        """
        # Start the simulator with a clean logical |0> state (encoded as |000> for rep code)
        # The encoding circuit is applied implicitly by starting at the encoded state for simplicity in Phase 1.
        self.current_state = self.simulator.execute_circuit(initial_state_str='0') # Logical |0> encoded and initialized in simulator

        self.current_step = 0
        
        # Initial observation for UES: includes noise config (general dict), QEC code props, and a simplified circuit (e.g. Identity op)
        # The NoiseEncoder in UES expects a dict, so we'll pass the dict from main_training_script.py
        # For Phase 2, NoiseEncoder should be updated to take NoiseModel object itself, or its dict representation.
        # For now, pass a representation of the Qiskit noise model as a simple dict.
        # This will be refined when NoiseEncoder in ues_model is updated for Phase 2.
        observation = {
            'noise_data': self.qiskit_noise_model.to_dict() if self.qiskit_noise_model else {}, # Pass NoiseModel dict representation
            'qec_code_props': { # These properties are static for Phase 1
                'code_type': 'repetition', 
                'distance': self.qec_code.distance, # Use actual distance from code obj
                'num_physical_qubits': self.qec_code.num_physical_qubits, 
                'num_logical_qubits': self.qec_code.num_logical_qubits
            },
            'circuit_ops': [{'type': 'gate', 'name': 'id', 'qubits': list(range(self.num_physical_qubits))}] # Represents current logical op (e.g., Logical Identity)
        }
        return observation

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Applies the chosen QEC strategy (action) and simulates one round of noisy logical operation + QEC.
        :param action: Dictionary of chosen QEC strategy (from UES output, e.g., integer indices).
                       e.g., {'mitigation_strategy': 0, 'decoder_choice': 2, 'compiler_action': 0}
        :return: (observation, reward, done, info)
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode

        # --- 1. Simulate a logical operation / idle step with inherent noise ---
        # The simulator object already has the Qiskit noise model configured internally.
        # We need to apply a logical operation (e.g., logical Identity) on `self.current_state`,
        # and noise will be automatically applied by `LogicalQubitSimulator.execute_circuit`.

        # Reset simulator's internal circuit builder and set its starting state for this step
        self.simulator.set_current_state(self.current_state) 
        
        # For Phase 1, we simulate a logical Identity operation
        # Add an identity gate for all physical qubits to simulate idle time with noise accumulation.
        # This acts as one "round" of noisy logical operation.
        for q_idx in range(self.num_physical_qubits):
             self.simulator.add_gate('id', [q_idx])

        # Execute this step's circuit (idle + noise) starting from `self.current_state`
        # noise is applied via the simulator's internal QiskitNoiseModel after gates.
        self.current_state = self.simulator.execute_circuit(initial_state_str=None, initial_state_dm=self.current_state.get_density_matrix())

        # --- 2. Perform Syndrome Extraction ---
        # Pass the current noisy state and the QEC code, along with the Qiskit noise model
        # if you want noise during measurement too (consistency).
        syndrome_str = extract_syndrome_from_state(self.current_state, self.qec_code, noise_model_aer=self.qiskit_noise_model)

        # --- 3. Apply Decoding and Correction based on Action ---
        # Map integer action back to string for decoder
        # These mappings need to match the output_decoders.py choices
        decoder_choices_map = {0: 'none_decoder', 1: 'basic_mwpm', 2: 'lookup_table'} 
        chosen_decoder_str = decoder_choices_map.get(action['decoder_choice'], 'none_decoder') 

        correction_pauli_str = 'III' # Default to no correction
        if chosen_decoder_str == 'lookup_table': # This maps to decode_three_qubit_repetition_code
            correction_pauli_str = decode_three_qubit_repetition_code(syndrome_str)
        # In Phase 2, you'll add more conditions for 'basic_mwpm' etc.

        # Apply correction to the quantum state using the simulator (without adding new noise for correction gates)
        if correction_pauli_str != 'III':
            # Reset simulator for correction application from current state
            self.simulator.set_current_state(self.current_state)
            
            # Convert correction_pauli_str ('XII', 'IXI', 'IIX') to gates on simulator
            if correction_pauli_str == 'XII':
                self.simulator.add_gate('x', [0])
            elif correction_pauli_str == 'IXI':
                self.simulator.add_gate('x', [1])
            elif correction_pauli_str == 'IIX':
                self.simulator.add_gate('x', [2])
            
            # Execute correction gates. Crucially, pass `qiskit_noise_model=None` to simulator for this execution
            # if you want ideal correction gates (no additional noise during correction application).
            # The simulator's __init__ takes `qiskit_noise_model`.
            # So, for ideal correction, temporarily override or create a new simulator.
            
            # Simpler approach: current simulator applies noise always.
            # For Phase 1, just accept noise on correction gates. Phase 2+ refines this.
            self.current_state = self.simulator.execute_circuit(initial_state_str=None, initial_state_dm=self.current_state.get_density_matrix())
            
        # --- 4. Calculate Reward ---
        # For 3-qubit rep code targeting logical |0>, ideal encoded state is |000>
        reward_fidelity = self.current_state.get_fidelity(self.ideal_encoded_state) 
        
        # Penalties for resources (simplified for Phase 1)
        physical_qubit_penalty = self.num_physical_qubits * 0.01 
        
        reward = reward_fidelity - physical_qubit_penalty
        
        # --- 5. Prepare next observation ---
        next_observation = {
            'noise_data': self.qiskit_noise_model.to_dict() if self.qiskit_noise_model else {}, # Still pass dict representation of noise
            'qec_code_props': { # QEC code props are static for Phase 1
                'code_type': 'repetition',
                'distance': self.qec_code.distance,
                'num_physical_qubits': self.qec_code.num_physical_qubits,
                'num_logical_qubits': self.qec_code.num_logical_qubits
            },
            'circuit_ops': [{'type': 'gate', 'name': 'id', 'qubits': list(range(self.num_physical_qubits))}] # Represents current logical op
        }
        
        info = {
            'syndrome': syndrome_str,
            'correction_applied': correction_pauli_str,
            'fidelity_after_step': reward_fidelity,
            'syndrome_detected_count': int(syndrome_str != '00') 
        }
        return next_observation, reward, done, info