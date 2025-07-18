# src/qera_core/logical_qubit_simulator.py
import numpy as np
# We will not use Qiskit for state evolution here, so these imports are removed from this file.
# from qiskit import QuantumCircuit, transpile 
# from qiskit_aer import AerSimulator 
# from qiskit_aer.noise import NoiseModel 
# from qiskit_aer.library import save_statevector, save_density_matrix 
from src.qera_core.state_representation import QuantumState 
# from qiskit.exceptions import QiskitError # No QiskitError caught here anymore.

# --- Define standard gate matrices (for NumPy simulation) ---
H_single = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
X_single = np.array([[0, 1], [1, 0]], dtype=complex)
Y_single = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_single = np.array([[1, 0], [0, -1]], dtype=complex)
I_single = np.identity(2, dtype=complex)

# --- Helper to get full system unitary (for NumPy simulation) ---
def _get_full_unitary(gate_matrix_single_or_two_qubit: np.ndarray, num_qubits: int, target_qubits: list, control_qubits: list = None) -> np.ndarray:
    if not target_qubits and (control_qubits is None or not control_qubits):
        raise ValueError("Target qubits or control/target for CNOT must be specified.")
    
    # Single-qubit gate
    if len(target_qubits) == 1 and (control_qubits is None or not control_qubits):
        target_q_idx = target_qubits[0]
        full_op = 1 
        for i in range(num_qubits):
            if i == target_q_idx:
                full_op = np.kron(full_op, gate_matrix_single_or_two_qubit) if isinstance(full_op, np.ndarray) else gate_matrix_single_or_two_qubit
            else:
                full_op = np.kron(full_op, I_single) if isinstance(full_op, np.ndarray) else I_single
        return full_op
    
    # CNOT gate (assuming 1 control and 1 target)
    elif control_qubits and len(control_qubits) == 1 and len(target_qubits) == 1:
        control_q_idx = control_qubits[0]
        target_q_idx = target_qubits[0]
        P0_control = np.array([[1, 0], [0, 0]], dtype=complex)
        P1_control = np.array([[0, 0], [0, 1]], dtype=complex)
        op_term0 = 1 
        for i in range(num_qubits):
            if i == control_q_idx: op_term0 = np.kron(op_term0, P0_control) if isinstance(op_term0, np.ndarray) else P0_control
            elif i == target_q_idx: op_term0 = np.kron(op_term0, I_single) if isinstance(op_term0, np.ndarray) else I_single
            else: op_term0 = np.kron(op_term0, I_single) if isinstance(op_term0, np.ndarray) else I_single
        op_term1 = 1 
        for i in range(num_qubits):
            if i == control_q_idx: op_term1 = np.kron(op_term1, P1_control) if isinstance(op_term1, np.ndarray) else P1_control
            elif i == target_q_idx: op_term1 = np.kron(op_term1, X_single) if isinstance(op_term1, np.ndarray) else X_single
            else: op_term1 = np.kron(op_term1, I_single) if isinstance(op_term1, np.ndarray) else I_single
        return op_term0 + op_term1
    else:
        raise ValueError("Gate type not supported or qubit indices incorrect for _get_full_unitary.")


class LogicalQubitSimulator:
    """
    Simulates quantum circuits with noise using pure NumPy.
    This version bypasses qiskit-aer for state evolution to ensure reliable state extraction.
    """
    def __init__(self, num_qubits: int, noise_config: dict = None, qiskit_noise_model=None): # noise_config re-introduced
        self.num_qubits = num_qubits
        self.noise_config = noise_config if noise_config is not None else {} # Re-introduce noise_config
        self.qiskit_noise_model = qiskit_noise_model # Keep this for future noise model generation
        
        self.operations = [] # List of (op_type, op_name, op_args) tuples for NumPy executor
        self.current_state = None 


    def add_gate(self, gate_name: str, target_qubits: list, control_qubits: list = None):
        """Adds a quantum gate operation to the internal operation queue."""
        self.operations.append(('gate', gate_name, target_qubits, control_qubits))

    def add_measurement(self, qubit_idx: int):
        """Adds a single qubit measurement operation to the circuit queue."""
        self.operations.append(('measure', qubit_idx))


    def execute_circuit(self, initial_state_str: str = '0', initial_state_dm: np.ndarray = None) -> QuantumState:
        """
        Executes the queued operations using pure NumPy.
        :param initial_state_str: Initial state string (e.g., '0' for |0...0>).
        :param initial_state_dm: Optional initial density matrix to directly set the state.
        :return: The final QuantumState object (density matrix).
        """
        # Ensure state_representation is imported here if not global
        from src.qera_core.state_representation import QuantumState
        from src.qera_core import noise_modeling # Re-import noise_modeling for NumPy noise

        # Initialize current_state for this execution, or use previous state
        if initial_state_dm is not None:
            self.current_state = QuantumState(self.num_qubits)
            self.current_state.set_density_matrix(initial_state_dm)
        elif initial_state_str:
            self.current_state = QuantumState(self.num_qubits, initial_state_str)
        elif self.current_state is None:
            self.current_state = QuantumState(self.num_qubits, '0') # Default to |0> if nothing else provided.
        
        # --- Execute operations with NumPy and apply noise ---
        for op_item in self.operations:
            op_type = op_item[0]
            
            if op_type == 'gate':
                gate_name = op_item[1]
                target_qubits = op_item[2]
                control_qubits = op_item[3] if len(op_item) > 3 else None

                # Get the full N-qubit unitary operator for the gate
                if gate_name == 'h': full_unitary = _get_full_unitary(H_single, self.num_qubits, target_qubits)
                elif gate_name == 'x': full_unitary = _get_full_unitary(X_single, self.num_qubits, target_qubits)
                elif gate_name == 'y': full_unitary = _get_full_unitary(Y_single, self.num_qubits, target_qubits)
                elif gate_name == 'z': full_unitary = _get_full_unitary(Z_single, self.num_qubits, target_qubits)
                elif gate_name == 'id': full_unitary = _get_full_unitary(I_single, self.num_qubits, target_qubits) # Identity gate
                elif gate_name == 'cx': full_unitary = _get_full_unitary(None, self.num_qubits, target_qubits, control_qubits)
                else: raise ValueError(f"Gate '{gate_name}' not supported in NumPy executor.")
                
                self.current_state.apply_unitary(full_unitary)
                
                # Apply noise *after* each gate application based on noise_config
                if 'depolarizing' in self.noise_config: # Assuming noise_config has 'depolarizing' key
                    noise_modeling.depolarizing_channel_per_qubit(
                        self.current_state, self.noise_config['depolarizing']
                    )
                if 'bit_flip' in self.noise_config: # Assuming noise_config has 'bit_flip' key
                    noise_modeling.bit_flip_channel_per_qubit(
                        self.current_state, self.noise_config['bit_flip']
                    )
                # More noise types here from noise_modeling.py can be added.
            
            elif op_type == 'measure':
                self.current_state.measure_qubit(op_item[1])
        
        self.operations = [] # Clear operations queue after execution
        return self.current_state

    def set_current_state(self, state: QuantumState):
        """Sets the simulator's current quantum state directly."""
        if not isinstance(state, QuantumState) or state.num_qubits != self.num_qubits:
            raise ValueError("Provided state is not a valid QuantumState object or has incorrect number of qubits.")
        self.current_state = state
        self.operations = [] # Clear operations queue

    def get_final_measurements(self, num_shots: int = 1) -> dict:
        """
        Simulates measuring all qubits multiple times in the Z-basis from the current_state.
        This now relies on QuantumState.measure_qubit for each shot.
        """
        counts = {}
        # Make a copy of the state for measurement to not modify the original
        initial_dm_for_measurement = self.current_state.get_density_matrix() 

        for _ in range(num_shots):
            # Create a temporary QuantumState for each shot to perform destructive measurement
            temp_state = QuantumState(self.num_qubits)
            temp_state.set_density_matrix(initial_dm_for_measurement)
            
            outcome_bits = ""
            for q_idx in range(self.num_qubits):
                outcome_bits += str(temp_state.measure_qubit(q_idx))
            
            counts[outcome_bits] = counts.get(outcome_bits, 0) + 1
        return counts


# Example usage (for internal testing/understanding - will be in notebooks)
if __name__ == "__main__":
    from src.qera_core import noise_modeling # Import local noise_modeling
    
    noise_config_test = {'depolarizing': 0.005, 'bit_flip': 0.001} # Use a simple noise config

    print("--- Simulating Bell State with NumPy and Noise ---")
    num_qubits_test = 2
    
    simulator = LogicalQubitSimulator(num_qubits_test, noise_config=noise_config_test)
    simulator.add_gate('h', [0])
    simulator.add_gate('cx', [0], [1]) 

    final_state = simulator.execute_circuit(initial_state_str='0') 
    print(f"Final Density Matrix (noisy Bell state):\n{final_state.get_density_matrix()}")

    ideal_bell_state_vector = 1/np.sqrt(2) * np.array([1, 0, 0, 1], dtype=complex)
    fidelity = final_state.get_fidelity(ideal_bell_state_vector)
    print(f"Fidelity with ideal Bell state: {fidelity:.4f}")

    counts = simulator.get_final_measurements(num_shots=1000)
    print(f"Measurement counts (1000 shots):\n{counts}")

    print("\n--- Simulating Single Qubit X-gate with Noise ---")
    sim_single = LogicalQubitSimulator(1, noise_config=noise_config_test)
    sim_single.add_gate('x', [0])
    final_state_single = sim_single.execute_circuit('0')
    print(f"Final Density Matrix (noisy |1>):\n{final_state_single.get_density_matrix()}")

    ideal_1_state = np.array([0, 1], dtype=complex)
    fidelity_single = final_state_single.get_fidelity(ideal_1_state)
    print(f"Fidelity with ideal |1> state: {fidelity_single:.4f}")