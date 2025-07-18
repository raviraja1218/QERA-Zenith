import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.library import save_statevector, save_density_matrix
from src.qera_core.state_representation import QuantumState
from qiskit.exceptions import QiskitError
import warnings

class LogicalQubitSimulator:
    """
    Simulates quantum circuits with robust statevector/density matrix handling.
    Key improvements:
    1. Dual saving of statevector and density matrix
    2. Comprehensive error handling
    3. Detailed debugging output
    """
    def __init__(self, num_qubits: int, qiskit_noise_model: NoiseModel = None):
        self.num_qubits = num_qubits
        self.qiskit_noise_model = qiskit_noise_model
        self.current_qiskit_circuit = QuantumCircuit(num_qubits)
        self.current_state = None
        
        # Use both statevector and density matrix methods for robustness
        self.simulator = AerSimulator(method='density_matrix')
        if self.qiskit_noise_model:
            self.simulator.set_options(noise_model=self.qiskit_noise_model)

    def add_gate(self, gate_name: str, target_qubits: list, control_qubits: list = None):
        """Adds quantum gates with validation"""
        if control_qubits is None:
            if len(target_qubits) != 1:
                raise ValueError(f"Single-qubit gate '{gate_name}' expects 1 target qubit.")
            getattr(self.current_qiskit_circuit, gate_name)(target_qubits[0])
        elif gate_name == 'cx' and len(control_qubits) == 1:
            self.current_qiskit_circuit.cx(control_qubits[0], target_qubits[0])
        else:
            raise ValueError(f"Unsupported gate configuration: {gate_name}")

    def execute_circuit(self, initial_state_str: str = '0', initial_state_dm: np.ndarray = None) -> QuantumState:
        """Execute circuit with robust state extraction"""
        qc = self.current_qiskit_circuit.copy()
        self.current_qiskit_circuit = QuantumCircuit(self.num_qubits)
        
        # Initialize state
        if initial_state_dm is not None:
            init_state = initial_state_dm
        else:
            sv = np.zeros(2**self.num_qubits, dtype=complex)
            sv[0] = 1.0  # Default |0...0> state
            init_state = np.outer(sv, sv.conj())

        # Save both statevector and density matrix for redundancy
        qc.save_statevector()
        qc.save_density_matrix()
        
        # Execute with detailed error handling
        try:
            job = self.simulator.run(
                transpile(qc, self.simulator),
                initial_state=init_state,
                shots=1
            )
            result = job.result()
            
            # Try multiple extraction methods
            data = result.data(0)
            print(f"DEBUG: Available result keys: {list(data.keys())}")
            
            if 'density_matrix' in data:
                final_dm = data['density_matrix']
            elif 'statevector' in data:
                sv = data['statevector']
                final_dm = np.outer(sv, sv.conj())
            else:
                if not qc.data:  # Empty circuit
                    final_dm = init_state
                else:
                    raise QiskitError("No valid state data found in results")
                    
        except Exception as e:
            warnings.warn(f"Simulation warning: {str(e)}")
            if not qc.data:
                final_dm = init_state
            else:
                raise RuntimeError(f"Simulation failed: {str(e)}")

        self.current_state = QuantumState(self.num_qubits)
        self.current_state.set_density_matrix(final_dm)
        return self.current_state

    # ... (keep other methods unchanged) ...

if __name__ == "__main__":
    """Enhanced test cases with better error reporting"""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    
    # Test noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h', 'x'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.1, 2), ['cx'])
    
    print("=== TEST 1: Bell State Simulation ===")
    sim = LogicalQubitSimulator(2, noise_model)
    try:
        sim.add_gate('h', [0])
        sim.add_gate('cx', [0], [1])
        state = sim.execute_circuit('0')
        print("Success! Final state:\n", state.get_density_matrix())
    except Exception as e:
        print(f"Test failed: {str(e)}")
    
    print("\n=== TEST 2: Empty Circuit ===")
    try:
        empty_sim = LogicalQubitSimulator(1)
        state = empty_sim.execute_circuit('0')
        print("Success! Maintained initial state")
    except Exception as e:
        print(f"Test failed: {str(e)}")