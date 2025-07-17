# src/qera_core/logical_qubit_simulator.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel # Correct import for Qiskit Aer's NoiseModel
from src.qera_core.state_representation import QuantumState # Your custom QuantumState class

# --- REMOVE all old numpy-based gate matrices like H_single, X_single, etc. from here ---
# --- REMOVE the old _get_full_unitary helper function from here ---
# These were from Phase 1 numpy simulator, now replaced by Qiskit's internal gate handling.

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
        
        # Internal Qiskit QuantumCircuit to build up operations for an execution step
        self.current_qiskit_circuit = QuantumCircuit(num_qubits) 
        
        # Your custom QuantumState object, representing the current density matrix
        self.current_state = None 

        # Set up the Aer simulator instance
        self.simulator = AerSimulator(method='density_matrix') # Ensure density_matrix method is used
        if self.qiskit_noise_model:
            self.simulator.set_options(noise_model=self.qiskit_noise_model)

    def add_gate(self, gate_name: str, target_qubits: list, control_qubits: list = None):
        """
        Adds a quantum gate operation to the internal Qiskit circuit queue.
        :param gate_name: Name of the gate (e.g., 'h', 'x', 'cx', 'id').
        :param target_qubits: List of integer indices of target qubits.
        :param control_qubits: Optional list of integer indices for control qubits (for multi-qubit gates).
        """
        if control_qubits is None or not control_qubits: # Single-qubit gates
            if len(target_qubits) != 1:
                raise ValueError(f"Single-qubit gate '{gate_name}' expects 1 target qubit, got {len(target_qubits)}.")
            q_idx = target_qubits[0]
            if gate_name == 'h': self.current_qiskit_circuit.h(q_idx)
            elif gate_name == 'x': self.current_qiskit_circuit.x(q_idx)
            elif gate_name == 'y': self.current_qiskit_circuit.y(q_idx)
            elif gate_name == 'z': self.current_qiskit_circuit.z(q_idx)
            elif gate_name == 'id': self.current_qiskit_circuit.id(q_idx) # Identity gate
            else: raise ValueError(f"Single-qubit gate '{gate_name}' not supported.")
        
        elif gate_name == 'cx' and len(control_qubits) == 1 and len(target_qubits) == 1:
            self.current_qiskit_circuit.cx(control_qubits[0], target_qubits[0])
        else:
            raise ValueError(f"Multi-qubit gate '{gate_name}' with controls/targets not supported or incorrect format.")

    def add_measurement(self, qubit_idx: int, classical_bit_idx: int = None):
        """
        Adds a single qubit measurement operation to the circuit queue.
        :param qubit_idx: Index of the qubit to measure.
        :param classical_bit_idx: Optional classical bit to store the result. If None, Qiskit creates one.
        """
        # Ensure classical bit exists if specified, create if not
        if classical_bit_idx is None:
            classical_bit_idx = qubit_idx 
            # Ensure enough classical bits are allocated in the circuit
            if classical_bit_idx >= self.current_qiskit_circuit.num_clbits:
                # Add a ClassicalRegister if current_qiskit_circuit doesn't have enough clbits
                self.current_qiskit_circuit.add_register(QuantumCircuit.ClassicalRegister(max(1, classical_bit_idx + 1)))
            
        self.current_qiskit_circuit.measure(qubit_idx, classical_bit_idx)


    def execute_circuit(self, initial_state_str: str = '0', initial_state_dm: np.ndarray = None) -> QuantumState:
        """
        Executes the queued Qiskit operations.
        :param initial_state_str: Initial state (e.g., '0' for |0...0>) string. Used if initial_state_dm is None.
        :param initial_state_dm: Optional initial density matrix to directly set the state.
        :return: The final QuantumState object (density matrix).
        """
        # Create a copy of the current Qiskit circuit to execute, then clear internal builder for next step
        qc_to_run = self.current_qiskit_circuit.copy() 
        self.current_qiskit_circuit = QuantumCircuit(self.num_qubits) # Reset internal builder for next step

        # Transpile the Qiskit circuit for the simulator
        transpiled_qc = transpile(qc_to_run, self.simulator)

        # Determine initial state for simulation run
        run_options = {'shots': 1} # For density_matrix method, shots only affect classical bit sampling
        
        if initial_state_dm is not None:
            # Use provided density matrix as initial state for Aer simulator
            run_options['initial_state'] = initial_state_dm
        elif initial_state_str:
            # Create initial statevector for Qiskit if string given (e.g., '0' for |00..0>)
            initial_sv = np.zeros(2**self.num_qubits, dtype=complex)
            if initial_state_str == '0': initial_sv[0] = 1.0
            else: raise ValueError("Only '0' string initial state supported when DM not provided for string initialization.")
            run_options['initial_state'] = initial_sv # For Aer, can also take statevector
        else:
            # If no initial state explicitly provided, and not set via set_current_state, assume |0...0>
            if self.current_state is None:
                initial_sv = np.zeros(2**self.num_qubits, dtype=complex)
                initial_sv[0] = 1.0
                run_options['initial_state'] = initial_sv
            else:
                # If current_state is already set (e.g., from env.step, set by set_current_state), use its DM
                run_options['initial_state'] = self.current_state.get_density_matrix()


        job = self.simulator.run(transpiled_qc, **run_options)
        result = job.result()
        
        # --- CRITICAL FIX FOR KeyError: 'density_matrix' ---
        final_qiskit_dm = None
        if 'density_matrix' in result.data():
            final_qiskit_dm = result.data()['density_matrix']
        elif 'statevector' in result.data():
            # If it's a statevector, convert it to a density matrix (|psi><psi|)
            sv = result.get_statevector() # Get the Statevector object from result
            final_qiskit_dm = np.outer(sv, sv.conj())
        else:
            raise KeyError("Neither 'density_matrix' nor 'statevector' found in Qiskit Aer result.data(). This might happen if simulation failed or result format changed.")
        # --- END FIX ---
        
        # Convert back to your internal QuantumState object
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
        # When setting current state, we need to reset the internal Qiskit circuit builder
        # with the correct number of classical bits if any measurements were added previously.
        # For simplicity, just reset with quantum bits. Classical bits can be added dynamically.
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
    # Using new Qiskit Aer imports
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
    simulator.add_gate('cx', [0], [1]) # CNOT control 0, target 1 (Corrected from [1],[0] to [0],[1] as per common Qiskit practice)

    final_state = simulator.execute_circuit(initial_state_str='0') # Start from |00>
    print(f"Final Density Matrix (noisy Bell state):\n{final_state.get_density_matrix()}")

    # Calculate fidelity with ideal Bell state |phi+> = 1/sqrt(2)(|00> + |11>)
    ideal_bell_state_vector = 1/np.sqrt(2) * np.array([1, 0, 0, 1], dtype=complex)
    fidelity = final_state.get_fidelity(ideal_bell_state_vector)
    print(f"Fidelity with ideal Bell state: {fidelity:.4f}")

    # Get measurement counts (will reflect readout errors)
    counts = simulator.get_final_measurements(num_shots=1000)
    print(f"Measurement counts (1000 shots):\n{counts}")

    print("\n--- Simulating Single Qubit X-gate with Noise ---")
    sim_single = LogicalQubitSimulator(1, qiskit_noise_model=noise_model_qiskit)
    sim_single.add_gate('x', [0]) # Apply X to |0> -> |1>
    final_state_single = sim_single.execute_circuit('0')
    print(f"Final Density Matrix (noisy |1>):\n{final_state_single.get_density_matrix()}")

    ideal_1_state = np.array([0, 1], dtype=complex)
    fidelity_single = final_state_single.get_fidelity(ideal_1_state)
    print(f"Fidelity with ideal |1> state: {fidelity_single:.4f}")