# src/qera_core/state_representation.py
import numpy as np

class QuantumState:
    """
    Represents a quantum state, internally as a density matrix (rho) to handle mixed states from noise.
    rho is a 2^N x 2^N complex matrix for N qubits.
    """
    def __init__(self, num_qubits: int, initial_state: str = '0'):
        """
        Initializes a quantum state.
        :param num_qubits: Number of qubits in the system.
        :param initial_state: '0' for |0...0> state, 'random' for a random pure state.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer.")

        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits # Size of state vector / one side of density matrix

        if initial_state == '0':
            # Start with |0...0> state, represented as a state vector
            initial_state_vector = np.zeros(self.dimension, dtype=complex)
            initial_state_vector[0] = 1.0
            # Convert to density matrix: |psi><psi|
            self.density_matrix = np.outer(initial_state_vector, initial_state_vector.conj())
        elif initial_state == 'random':
            # Generate a random pure state and convert to density matrix
            random_state_vector = np.random.rand(self.dimension) + 1j * np.random.rand(self.dimension)
            random_state_vector = random_state_vector / np.linalg.norm(random_state_vector) # Normalize
            self.density_matrix = np.outer(random_state_vector, random_state_vector.conj())
        else:
            raise ValueError("Initial state not recognized. Use '0' or 'random'.")

    def get_density_matrix(self) -> np.ndarray:
        """Returns the current density matrix."""
        return self.density_matrix

    def set_density_matrix(self, dm: np.ndarray):
        """Sets the quantum state directly from a given density matrix."""
        if dm.shape != (self.dimension, self.dimension) or not np.allclose(np.trace(dm), 1.0):
            raise ValueError("Invalid density matrix shape or trace. Must be 2^N x 2^N and trace to 1.")
        self.density_matrix = dm

    def apply_unitary(self, unitary_matrix: np.ndarray):
        """
        Applies a unitary operation (gate) to the density matrix.
        rho -> U rho U^dagger
        :param unitary_matrix: The N-qubit unitary matrix (2^N x 2^N).
        """
        if unitary_matrix.shape != (self.dimension, self.dimension):
            raise ValueError(f"Unitary matrix dimension {unitary_matrix.shape} mismatch with system dimension {self.dimension}.")

        self.density_matrix = unitary_matrix @ self.density_matrix @ unitary_matrix.conj().T

    def apply_kraus_operators(self, kraus_ops: list[np.ndarray]):
        """
        Applies a quantum operation defined by a set of Kraus operators.
        rho -> sum_k M_k rho M_k^dagger
        :param kraus_ops: A list of NumPy arrays, each representing a Kraus operator M_k.
                          Each M_k must be 2^N x 2^N.
        """
        new_density_matrix = np.zeros_like(self.density_matrix, dtype=complex)
        for M_k in kraus_ops:
            if M_k.shape != (self.dimension, self.dimension):
                raise ValueError(f"Kraus operator M_k has invalid shape {M_k.shape}. Expected ({self.dimension}, {self.dimension}).")
            new_density_matrix += M_k @ self.density_matrix @ M_k.conj().T
        self.density_matrix = new_density_matrix

    def measure_qubit(self, qubit_idx: int) -> int:
        """
        Simulates measurement of a single qubit in the Z-basis.
        Updates the state (density matrix) based on the measurement outcome.
        :param qubit_idx: The index of the qubit to measure (0-indexed).
        :return: The measurement outcome (0 or 1).
        """
        if not 0 <= qubit_idx < self.num_qubits:
            raise ValueError("Qubit index out of bounds.")

        # Projectors for outcome 0 and 1 on the measured qubit
        P0_single = np.array([[1, 0], [0, 0]], dtype=complex)
        P1_single = np.array([[0, 0], [0, 1]], dtype=complex)

        # Construct the full system projectors (I x ... x P0/1 x ... x I)
        P0_full = 1 # Initialize for Kronecker product chain
        P1_full = 1 
        I_2 = np.identity(2, dtype=complex) # Identity for non-measured qubits

        for i in range(self.num_qubits):
            if i == qubit_idx:
                P0_full = np.kron(P0_full, P0_single) if isinstance(P0_full, np.ndarray) else P0_single
                P1_full = np.kron(P1_full, P1_single) if isinstance(P1_full, np.ndarray) else P1_single
            else:
                P0_full = np.kron(P0_full, I_2) if isinstance(P0_full, np.ndarray) else I_2
                P1_full = np.kron(P1_full, I_2) if isinstance(P1_full, np.ndarray) else I_2

        # Calculate probabilities of outcomes
        prob_0 = np.trace(P0_full @ self.density_matrix).real
        prob_1 = np.trace(P1_full @ self.density_matrix).real

        # Due to numerical errors, probabilities might be slightly off. Normalize and clip.
        total_prob = prob_0 + prob_1
        if total_prob > 1e-9: # Avoid division by zero for very small probabilities
            prob_0 /= total_prob
            prob_1 /= total_prob
        else: # Fallback for extremely low total probability, should ideally not happen
            prob_0, prob_1 = 0.5, 0.5 

        # Ensure probabilities are within [0, 1] for np.random.choice
        prob_0 = np.clip(prob_0, 0.0, 1.0)
        prob_1 = np.clip(prob_1, 0.0, 1.0)
        # Re-normalize if clipping changed sum significantly
        current_sum = prob_0 + prob_1
        if current_sum > 0:
            prob_0 /= current_sum
            prob_1 /= current_sum
        else: # If both are zero after clip, assign equal prob
            prob_0, prob_1 = 0.5, 0.5


        outcome = np.random.choice([0, 1], p=[prob_0, prob_1])

        # Update the state based on measurement outcome (post-measurement state)
        # rho_prime = (P_outcome * rho * P_outcome) / Prob_outcome
        if outcome == 0:
            self.density_matrix = (P0_full @ self.density_matrix @ P0_full.conj().T) / prob_0 if prob_0 > 1e-9 else np.zeros_like(self.density_matrix)
        else:
            self.density_matrix = (P1_full @ self.density_matrix @ P1_full.conj().T) / prob_1 if prob_1 > 1e-9 else np.zeros_like(self.density_matrix)

        # Ensure density matrix remains valid (trace to 1, Hermitian) due to numerical errors
        self.density_matrix /= np.trace(self.density_matrix) # Re-normalize
        self.density_matrix = (self.density_matrix + self.density_matrix.conj().T) / 2 # Ensure Hermitian

        return outcome

    def get_fidelity(self, target_state_vector: np.ndarray) -> float:
        """
        Calculates the fidelity of the current density matrix with a target pure state vector.
        F = sqrt(<psi_target|rho|psi_target>) (pure state fidelity)
        :param target_state_vector: The state vector of the ideal pure state to compare against.
        :return: Fidelity as a float between 0 and 1.
        """
        if target_state_vector.shape[0] != self.dimension:
            raise ValueError("Target state vector dimension mismatch.")

        # Fidelity of a mixed state rho with a pure state |psi> is sqrt(<psi|rho|psi>)
        # This is equivalent to F(rho, |psi><psi|) = Tr(sqrt(sqrt(rho) |psi><psi| sqrt(rho)))
        # which simplifies to sqrt(<psi|rho|psi>) for pure target.
        fidelity_squared = (target_state_vector.conj().T @ self.density_matrix @ target_state_vector).real

        # Ensure fidelity_squared is not negative due to numerical errors
        return np.sqrt(max(0, fidelity_squared))

    def __str__(self):
        return f"Quantum State ({self.num_qubits} qubits):\n{self.density_matrix}"