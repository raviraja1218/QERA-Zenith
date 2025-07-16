# src/qera_core/noise_modeling.py
import numpy as np
from src.qera_core.state_representation import QuantumState # Corrected import path

# Define common Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def _get_single_qubit_kraus_full_system(kraus_op_single: np.ndarray, num_qubits: int, target_qubit: int) -> np.ndarray:
    """
    Helper to build an N-qubit Kraus operator for a single-qubit operation.
    Constructs a (2^N x 2^N) operator by tensoring kraus_op_single at target_qubit
    with identity matrices for other qubits.
    """
    op = 1
    for i in range(num_qubits):
        if i == target_qubit:
            op = np.kron(op, kraus_op_single) if isinstance(op, np.ndarray) else kraus_op_single
        else:
            op = np.kron(op, I) if isinstance(op, np.ndarray) else I
    return op

def depolarizing_channel_per_qubit(state: QuantumState, p_error: float):
    """
    Applies a depolarizing channel independently to each qubit in the system.
    rho -> (1-p)rho + p/3(XrhoX + YrhoY + ZrhoZ) for each qubit.
    This is a standard depolarizing channel where p_error is the probability
    of the *error event* (not the probability of each Pauli error).
    :param state: QuantumState object (density matrix).
    :param p_error: Error rate for *each* qubit (between 0 and 1, usually p < 3/4).
    """
    if not 0 <= p_error <= 1:
        raise ValueError("Error rate must be between 0 and 1.")
    if p_error == 0:
        return # No error to apply

    # Kraus operators for single qubit depolarizing
    M0_single = np.sqrt(1 - 3*p_error/4) * I
    Mx_single = np.sqrt(p_error/4) * X
    My_single = np.sqrt(p_error/4) * Y
    Mz_single = np.sqrt(p_error/4) * Z

    single_qubit_kraus_set = [M0_single, Mx_single, My_single, Mz_single]

    # To apply depolarizing independently to N qubits, the total set of Kraus operators
    # for the N-qubit system is the tensor product of all single-qubit Kraus sets.
    # E.g., for 2 qubits: M00, M0X, M0Y, M0Z, MX0, MXX, MXY, MXZ, etc. (16 operators)
    # This grows as 4^N, quickly becoming too many.

    # For Phase 1, we will approximate independent application by applying
    # the single-qubit depolarizing channel *sequentially* to each qubit.
    # This is an approximation and will be replaced by `qiskit.providers.aer.noise.NoiseModel` in Phase 2
    # for proper multi-qubit noise.

    for q_idx in range(state.num_qubits):
        kraus_ops_for_current_qubit = [
            _get_single_qubit_kraus_full_system(M_single, state.num_qubits, q_idx)
            for M_single in single_qubit_kraus_set
        ]
        state.apply_kraus_operators(kraus_ops_for_current_qubit)

def bit_flip_channel_per_qubit(state: QuantumState, p_error: float):
    """
    Applies a bit-flip channel (probabilistic Pauli-X) independently to each qubit.
    rho -> (1-p)rho + p XrhoX
    :param state: QuantumState object (density matrix).
    :param p_error: Probability of bit flip for each qubit.
    """
    if not 0 <= p_error <= 1:
        raise ValueError("Error rate must be between 0 and 1.")
    if p_error == 0:
        return # No error to apply

    M0_single = np.sqrt(1 - p_error) * I
    M1_single = np.sqrt(p_error) * X # X as the error operation

    single_qubit_kraus_set = [M0_single, M1_single]

    for q_idx in range(state.num_qubits):
        kraus_ops_for_current_qubit = [
            _get_single_qubit_kraus_full_system(M_single, state.num_qubits, q_idx)
            for M_single in single_qubit_kraus_set
        ]
        state.apply_kraus_operators(kraus_ops_for_current_qubit)