# src/qera_core/qec_codes/code_definitions.py
import numpy as np
from qiskit import QuantumCircuit # Use Qiskit for easy circuit building

class QECCode:
    """Base class for Quantum Error Correction Codes."""
    def __init__(self, num_logical_qubits: int, num_physical_qubits: int):
        self.num_logical_qubits = num_logical_qubits
        self.num_physical_qubits = num_physical_qubits
        self.stabilizer_generators = [] # List of Pauli strings or operators
        self.encoding_circuit = None # Qiskit QuantumCircuit
        self.syndrome_measurement_circuit = None # Qiskit QuantumCircuit

    def get_encoding_circuit(self) -> QuantumCircuit:
        raise NotImplementedError("Encoding circuit not implemented for base class.")

    def get_syndrome_measurement_circuit(self) -> QuantumCircuit:
        raise NotImplementedError("Syndrome measurement circuit not implemented for base class.")

    def get_stabilizer_generators(self) -> list[str]:
        """Returns list of stabilizer generators as Pauli strings (e.g., 'ZII', 'IZI')."""
        return self.stabilizer_generators

class ThreeQubitRepetitionCode(QECCode): # <--- THIS IS THE CLASS NAME IT'S LOOKING FOR
    """
    The 3-qubit bit-flip repetition code for a single logical qubit.
    Logical |0> = |000>, Logical |1> = |111>
    Stabilizer generators: Z_0 Z_1, Z_1 Z_2
    """
    def __init__(self):
        super().__init__(num_logical_qubits=1, num_physical_qubits=3)
        self.stabilizer_generators = ["ZII", "IZI"] # Simplified for definition
        # Qiskit representation for these (Z Z I, I Z Z)

    def get_encoding_circuit(self) -> QuantumCircuit:
        """
        Returns the Qiskit QuantumCircuit to encode |q> -> |q_L>.
        Logical |0> = |000>, Logical |1> = |111>
        Encoding: H(0), CNOT(0,1), CNOT(0,2) applied to |000> yields (|000>+|111>)/sqrt(2)
        If input is |psi> = a|0> + b|1>, output is a|000> + b|111>
        """
        qc = QuantumCircuit(self.num_physical_qubits, self.num_physical_qubits - self.num_logical_qubits) # 3 phys, 2 classical bits for syndrome
        qc.cx(0, 1)
        qc.cx(0, 2)
        self.encoding_circuit = qc
        return qc

    def get_syndrome_measurement_circuit(self) -> QuantumCircuit:
        """
        Returns the Qiskit QuantumCircuit for syndrome measurement.
        Measures Z_0 Z_1 and Z_1 Z_2.
        Requires 2 ancilla qubits for measurement (total 5 qubits: 3 data, 2 ancilla).
        """
        # This circuit is for actual measurement during QEC
        # It takes the 3 data qubits, adds 2 ancillas, and measures stabilizers.
        # For Phase 1, we might simulate error, then manually derive syndrome.

        # A common pattern for syndrome extraction
        qc = QuantumCircuit(self.num_physical_qubits + 2, 2) # 3 data + 2 ancilla, 2 classical
        # Measure Z0Z1
        qc.cx(0, 3) # Data 0 to Ancilla 0
        qc.cx(1, 3) # Data 1 to Ancilla 0
        qc.measure(3, 0) # Measure Ancilla 0 to classical bit 0

        # Measure Z1Z2
        qc.cx(1, 4) # Data 1 to Ancilla 1
        qc.cx(2, 4) # Data 2 to Ancilla 1
        qc.measure(4, 1) # Measure Ancilla 1 to classical bit 1

        self.syndrome_measurement_circuit = qc
        return qc