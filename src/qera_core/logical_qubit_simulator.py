# Cell 3.0 or 3.1: Code for src/qera_core/logical_qubit_simulator.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel 
from src.qera_core.state_representation import QuantumState 
from qiskit.exceptions import QiskitError # Import QiskitError for robust error handling

class LogicalQubitSimulator:
    """
    Simulates quantum circuits, including noise, by leveraging Qiskit Aer.
    This simulator builds a Qiskit QuantumCircuit internally and executes it with AerSimulator.
    """
    def __init__(self, num_qubits: int, qiskit_noise_model: NoiseModel = None):
        """
        Initializes the simulator with a specified number of qubits and an optional Qiskit Aer NoiseModel.
        :param num_qubits: Total number of physical qubits in the simulation.
        :param qiskit_noise_model: A qiskit_aer.noise.NoiseModel object. If None, assumes ideal simulation.
        """
        self.num_qubits = num_qubits
        self.qiskit_noise_model = qiskit_noise_model
        
        self.current_qiskit_circuit = QuantumCircuit(num_qubits) 
        
        self.current_state = None 

        self.simulator = AerSimulator(method='density_matrix') 
        if self.qiskit_noise_model:
            self.simulator.set_options(noise_model=self.qiskit_noise_model)

    def add_gate(self, gate_name: str, target_qubits: list, control_qubits: list = None):
        """
        Adds a quantum gate operation to the internal Qiskit circuit queue.
        :param gate_name: Name of the gate (e.g., 'h', 'x', 'cx', 'id').
        :param target_qubits: List of integer indices of target qubits.
        :param control_qubits: Optional list of integer indices for control qubits (for multi-qubit gates).
        """
        if control_qubits is None or not control_qubits: 
            if len(target_qubits) != 1:
                raise ValueError(f"Single-qubit gate '{gate_name}' expects 1 target qubit, got {len(target_qubits)}.")
            q_idx = target_qubits[0]
            if gate_name == 'h': self.current_qiskit_circuit.h(q_idx)
            elif gate_name == 'x': self.current_qiskit_circuit.x(q_idx)
            elif gate_name == 'y': self.current_qiskit_circuit.y(q_idx)
            elif gate_name == 'z': self.current_qiskit_circuit.z(q_idx)
            elif gate_name == 'id': self.current_qiskit_circuit.id(q_idx) 
            else: raise ValueError(f"Single-qubit gate '{gate_name}' not supported.")
        
        elif gate_name == 'cx' and len(control_qubits) == 1 and len(target_qubits) == 1:
            self.current_qiskit_circuit.cx(control_qubits[0], target_qubits[0])
        else:
            raise ValueError(f"Multi-qubit gate '{gate_name}' with controls/targets not supported or incorrect format.")

    def add_measurement(self, qubit_idx: int, classical_bit_idx: int = None):
        if classical_bit_idx is None:
            classical_bit_idx = qubit_idx 
            if classical_bit_idx >= self.current_qiskit_circuit.num_clbits:
                self.current_qiskit_circuit.add_register(QuantumCircuit.ClassicalRegister(max(1, classical_bit_idx + 1)))
            
        self.current_qiskit_circuit.measure(qubit_idx, classical_bit_idx)


    def execute_circuit(self, initial_state_str: str = '0', initial_state_dm: np.ndarray = None) -> QuantumState:
        qc_to_run = self.current_qiskit_circuit.copy() 
        self.current_qiskit_circuit = QuantumCircuit(self.num_qubits) 

        transpiled_qc = transpile(qc_to_run, self.simulator)

        run_options = {'shots': 1} 
        
        # Determine the effective initial state for debugging and fallback
        effective_initial_dm = None
        if initial_state_dm is not None:
            effective_initial_dm = initial_state_dm
        elif initial_state_str:
            initial_sv = np.zeros(2**self.num_qubits, dtype=complex)
            if initial_state_str == '0': initial_sv[0] = 1.0
            else: raise ValueError("Only '0' string initial state supported when DM not provided for string initialization.")
            effective_initial_dm = np.outer(initial_sv, initial_sv.conj()) # Convert SV to DM
        else:
            if self.current_state is not None:
                effective_initial_dm = self.current_state.get_density_matrix()
            else: # Default to |0> if no state provided anywhere
                initial_sv = np.zeros(2**self.num_qubits, dtype=complex)
                initial_sv[0] = 1.0
                effective_initial_dm = np.outer(initial_sv, initial_sv.conj())

        # Always pass initial_state to simulator.run()
        run_options['initial_state'] = effective_initial_dm


        job = self.simulator.run(transpiled_qc, **run_options)
        result = job.result()
        
        # --- ULTIMATE ROBUST FIX FOR QISKIT AER RESULT EXTRACTION (FINAL FINAL VERSION!) ---
        final_qiskit_dm = None
        
        print(f"DEBUG: Qiskit Aer job.result().data() contains keys: {list(result.data().keys())}") # DEBUG PRINT
        
        try:
            # Priority 1: Try to get density matrix via result.get_density_matrix()
            # Check if method exists before calling, then attempt.
            if hasattr(result, 'get_density_matrix'):
                final_qiskit_dm = result.get_density_matrix() 
                print("DEBUG: Successfully retrieved density matrix using result.get_density_matrix().")
            else:
                raise AttributeError("Attribute get_density_matrix is not defined for result object.") # Raise to go to next except block
        except AttributeError as e_dm_attr: # Catch AttributeError specifically for get_density_matrix
            print(f"DEBUG: result.get_density_matrix() failed ({e_dm_attr}). Trying result.get_statevector().")
            try:
                # Priority 2: Try to get statevector and convert
                if hasattr(result, 'get_statevector'):
                    sv = result.get_statevector()
                    final_qiskit_dm = np.outer(sv, sv.conj())
                    print("DEBUG: Successfully retrieved statevector and converted to density matrix.")
                else:
                    raise AttributeError("Attribute get_statevector is not defined for result object.") # Raise to go to next except block
            except AttributeError as e_sv_attr: # Catch AttributeError for statevector
                print(f"DEBUG: result.get_statevector() failed ({e_sv_attr}). Checking if circuit was empty or had no operations.")
                # Priority 3: Fallback if circuit had no operations (like in env.reset() or just initial state setup)
                # Check if the Qiskit circuit built (qc_to_run) has any actual operations/instructions in its data list.
                if not qc_to_run.data: # .data is the list of instructions in a Qiskit circuit
                    print("DEBUG: Circuit was empty (no operations added). Falling back to initial state DM.")
                    final_qiskit_dm = effective_initial_dm # Use the initial state passed to simulator.run()
                else:
                    # If it had operations but still no state data, then a genuine simulation problem.
                    raise KeyError(
                        f"Neither valid density matrix nor statevector found in Qiskit Aer result for non-empty circuit. "
                        f"Last attempts errors: DM attr ({e_dm_attr}), SV attr ({e_sv_attr}). "
                        f"This might indicate simulation failure or unexpected result format."
                    )
        # --- END FIX ---
        
        self.current_state = QuantumState(self.num_qubits)
        self.current_state.set_density_matrix(final_qiskit_dm)

        return self.current_state

    def set_current_state(self, state: QuantumState):
        """
        Sets the simulator's current quantum state directly.
        This is useful for continuing a simulation from a specific state,
        e.g., passing the state from one step to the next in an RL environment.
        It also clears the Qiskit circuit queue.
        :param state: The QuantumState object to set as the current state.
        """
        if not isinstance(state, QuantumState) or state.num_qubits != self.num_qubits:
            raise ValueError("Provided state is not a valid QuantumState object or has incorrect number of qubits.")
        self.current_state = state
        self.current_qiskit_circuit = QuantumCircuit(self.num_qubits) 


    def get_final_measurements(self, num_shots: int = 1) -> dict:
        """
        Simulates measuring all qubits multiple times in the Z-basis from the current_state.
        Uses Qiskit Aer's measurement capability on the current density matrix.
        :param num_shots: Number of times to repeat the measurement.
        :return: A dictionary of measurement outcomes (e.g., {'00': 50, '01': 25, ...}).
        """
        # Create a temporary Qiskit circuit just for measurements
        # Ensure enough classical bits for all qubits to be measured
        meas_qc = QuantumCircuit(self.num_qubits, self.num_qubits) # Qiskit automatically creates clbits here
        meas_qc.measure(range(self.num_qubits), range(self.num_qubits))

        # Transpile the measurement circuit
        transpiled_meas_qc = transpile(meas_qc, self.simulator)

        # Run simulation starting from current_state density matrix
        job = self.simulator.run(transpiled_meas_qc, shots=num_shots, initial_state=self.current_state.get_density_matrix())
        result = job.result()
        counts = result.get_counts(transpiled_meas_qc)
        return counts

# Example usage (for internal testing/understanding - will be in notebooks)
if __name__ == "__main__":
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError 
    
    # Create a simple Qiskit Aer noise model for testing
    noise_model_qiskit = NoiseModel()
    
    # Add a 1-qubit depolarizing error to all H and X gates
    depol_error_1q = depolarizing_error(0.01, 1) # 1% depolarizing on 1-qubit gates
    noise_model_qiskit.add_all_qubit_quantum_error(depol_error_1q, ['h', 'x'])

    # Add a 2-qubit depolarizing error to CX gates
    depol_error_2q = depolarizing_error(0.02, 2) # 2% depolarizing on 2-qubit gates
    noise_model_qiskit.add_all_qubit_quantum_error(depol_error_2q, ['cx'])

    # Add a basic readout error (e.g., 5% probability of error for each readout)
    readout_error = ReadoutError([[0.95, 0.05], [0.05, 0.95]])
    noise_model_qiskit.add_all_qubit_readout_error(readout_error)

    print("--- Simulating Bell State with Qiskit Aer NoiseModel ---")
    num_qubits_test = 2
    
    # Initialize simulator with the Qiskit NoiseModel
    simulator = LogicalQubitSimulator(num_qubits_test, qiskit_noise_model=noise_model_qiskit)
    
    # Build a Bell state circuit
    simulator.add_gate('h', [0])
    simulator.add_gate('cx', [0], [1]) 

    final_state = simulator.execute_circuit(initial_state_str='0') 
    print(f"Final Density Matrix (noisy Bell state):\n{final_state.get_density_matrix()}")

    ideal_bell_state_vector = 1/np.sqrt(2) * np.array([1, 0, 0, 1], dtype=complex)
    fidelity = final_state.get_fidelity(ideal_bell_state_vector)
    print(f"Fidelity with ideal Bell state: {fidelity:.4f}")

    # Get measurement counts (will reflect readout errors)
    counts = simulator.get_final_measurements(num_shots=1000)
    print(f"Measurement counts (1000 shots):\n{counts}")

    print("\n--- Simulating Single Qubit X-gate with Noise ---")
    sim_single = LogicalQubitSimulator(1, qiskit_noise_model=noise_model_qiskit)
    sim_single.add_gate('x', [0])
    final_state_single = sim_single.execute_circuit('0')
    print(f"Final Density Matrix (noisy |1>):\n{final_state_single.get_density_matrix()}")

    ideal_1_state = np.array([0, 1], dtype=complex)
    fidelity_single = final_state_single.get_fidelity(ideal_1_state)
    print(f"Fidelity with ideal |1> state: {fidelity_single:.4f}")