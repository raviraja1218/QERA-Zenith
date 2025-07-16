# src/qera_core/qec_codes/syndrome_extraction.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix # Still needed for constructing the DensityMatrix object from numpy array

from src.qera_core.state_representation import QuantumState
from src.qera_core.qec_codes.code_definitions import QECCode 

def extract_syndrome_from_state(
    noisy_state: QuantumState, 
    qec_code: QECCode, 
    noise_model_aer=None
) -> str:
    """
    Extracts the classical syndrome from a noisy quantum state using Qiskit Aer.
    For Phase 1, focus on a 3-qubit repetition code.
    :param noisy_state: The current QuantumState object (density matrix) of data qubits.
    :param qec_code: The QECCode instance (e.g., ThreeQubitRepetitionCode).
    :param noise_model_aer: Optional Qiskit Aer noise model for simulation (for Phase 2+).
    :return: A binary string representing the syndrome (e.g., '00', '01').
    """
    if not isinstance(qec_code, QECCode):
        raise TypeError("qec_code must be an instance of QECCode or its subclass.")

    syndrome_qc = qec_code.get_syndrome_measurement_circuit()
    
    # Prepare the AerSimulator, explicitly using 'density_matrix' method
    simulator = AerSimulator(method='density_matrix')

    if noise_model_aer:
        simulator.set_options(noise_model=noise_model_aer)
    
    # Transpile the syndrome measurement circuit for the simulator.
    # IMPORTANT: Do NOT initialize the circuit with the density matrix here using .initialize() or .prepare_state().
    # Instead, we pass the initial density matrix directly to the simulator.run() call.
    transpiled_qc = transpile(syndrome_qc, simulator)

    # Run the simulation, providing the initial density matrix directly to the `initial_state` argument.
    # This is the correct way to start an AerSimulator (density_matrix method) with a given density matrix.
    job = simulator.run(
        transpiled_qc, 
        shots=1, 
        initial_state=noisy_state.get_density_matrix() # Pass the NumPy array directly
    )
    result = job.result()
    counts = result.get_counts(transpiled_qc)

    # --- Syndrome String Extraction Logic (Remains as is) ---
    # The counts dict keys are like '00', '01' (classical bits from measurement).
    # Since we are measuring ancillas and mapping them to classical bits,
    # the raw key from `counts` is usually the syndrome string directly.
    
    if not counts: # Handle case where counts might be empty if simulation fails
        # This should ideally not happen for shots=1 if execution is successful
        print("Warning: No counts obtained from syndrome measurement. Returning '00' (no syndrome) as default.")
        return '00'
        
    syndrome_outcome = list(counts.keys())[0] # Get the single outcome string
    syndrome_str = syndrome_outcome # Assuming counts keys are direct syndrome string '00', '01' etc.
    
    return syndrome_str