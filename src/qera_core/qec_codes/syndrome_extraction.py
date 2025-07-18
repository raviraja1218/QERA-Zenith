# Cell X (e.g., Cell 3.X): Code for src/qera_core/qec_codes/syndrome_extraction.py
import numpy as np
# REMOVED Qiskit Aer/Core imports as we are using NumPy simulator now for this function
# from qiskit import QuantumCircuit, transpile 
# from qiskit_aer import AerSimulator
# from qiskit.quantum_info import DensityMatrix 
# from qiskit.exceptions import QiskitError

from src.qera_core.state_representation import QuantumState
from src.qera_core.qec_codes.code_definitions import QECCode 
from src.qera_core.logical_qubit_simulator import LogicalQubitSimulator # <-- NEW IMPORT: Use your own simulator


def extract_syndrome_from_state(
    noisy_state: QuantumState, 
    qec_code: QECCode, 
    # REMOVED: noise_model_aer=None # No longer using Aer's noise model for this directly
) -> str:
    """
    Extracts the classical syndrome from a noisy quantum state using YOUR custom NumPy simulator.
    For Phase 1, focus on a 3-qubit repetition code.
    :param noisy_state: The current QuantumState object (density matrix) of data qubits.
    :param qec_code: The QECCode instance (e.g., ThreeQubitRepetitionCode).
    :return: A binary string representing the syndrome (e.g., '00', '01').
    """
    if not isinstance(qec_code, QECCode):
        raise TypeError("qec_code must be an instance of QECCode or its subclass.")

    # --- Simulate syndrome measurement circuit using YOUR NumPy simulator ---
    # The 3-qubit repetition code has 3 data qubits and uses 2 ancilla qubits (total 5 qubits for measurement)
    # The syndrome measurement circuit (e.g., from code_definitions.py) provides the CNOT sequence.
    # We will simulate this CNOT sequence on a temporary simulator.
    
    num_data_qubits = qec_code.num_physical_qubits
    num_ancilla_qubits = 2 # For 3-qubit repetition code
    total_qubits_for_syndrome_sim = num_data_qubits + num_ancilla_qubits

    # Create a temporary LogicalQubitSimulator instance for just this measurement
    syndrome_sim = LogicalQubitSimulator(total_qubits_for_syndrome_sim) 
    
    # Initialize the data qubits with the current noisy_state
    # Ancillas (qubits at num_data_qubits and onwards) are implicitly initialized to |0>
    syndrome_sim.set_current_state(noisy_state) 
    
    # Add the CNOT operations for syndrome extraction (Hardcoded for 3-qubit rep code for Phase 1)
    # Syndrome 0: Z0Z1 (measured by ancilla q3): CX(0,3), CX(1,3)
    syndrome_sim.add_gate('cx', [0], [3]) # Control data_q0, Target ancilla_q3
    syndrome_sim.add_gate('cx', [1], [3]) # Control data_q1, Target ancilla_q3

    # Syndrome 1: Z1Z2 (measured by ancilla q4): CX(1,4), CX(2,4)
    syndrome_sim.add_gate('cx', [1], [4]) # Control data_q1, Target ancilla_q4
    syndrome_sim.add_gate('cx', [2], [4]) # Control data_q2, Target ancilla_q4

    # Execute the syndrome circuit (no noise during measurement for simplicity)
    final_syndrome_state = syndrome_sim.execute_circuit(initial_state_str=None, initial_state_dm=syndrome_sim.current_state.get_density_matrix())

    # Measure ancilla qubits (at indices 3 and 4) to get syndrome bits
    outcome_ancilla0 = final_syndrome_state.measure_qubit(3) 
    outcome_ancilla1 = final_syndrome_state.measure_qubit(4) 
    
    syndrome_str = f"{outcome_ancilla0}{outcome_ancilla1}" # e.g., '00', '01', '10', '11'

    return syndrome_str