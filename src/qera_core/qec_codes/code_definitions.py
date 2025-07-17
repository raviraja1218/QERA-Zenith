# Cell X (e.g., Cell 3.X): Code for src/qera_core/qec_codes/code_definitions.py
import numpy as np
from qiskit import QuantumCircuit 
from qiskit.quantum_info import Pauli # <-- NEW IMPORT: Needed for stabilizer definitions

class QECCode:
    """Base class for Quantum Error Correction Codes."""
    def __init__(self, num_logical_qubits: int, num_physical_qubits: int):
        self.num_logical_qubits = num_logical_qubits
        self.num_physical_qubits = num_physical_qubits
        self.stabilizer_generators = [] # List of Pauli objects or strings
        self.encoding_circuit = None # Qiskit QuantumCircuit
        self.syndrome_measurement_circuit = None # Qiskit QuantumCircuit

    def get_encoding_circuit(self) -> QuantumCircuit:
        raise NotImplementedError("Encoding circuit not implemented for base class.")

    def get_syndrome_measurement_circuit(self) -> QuantumCircuit:
        raise NotImplementedError("Syndrome measurement circuit not implemented for base class.")

    def get_stabilizer_generators(self) -> list[Pauli]: # <-- UPDATED TYPE HINT: Returns list of Qiskit Pauli objects
        """Returns list of stabilizer generators as Qiskit Pauli objects."""
        return self.stabilizer_generators

class ThreeQubitRepetitionCode(QECCode): 
    """
    The 3-qubit bit-flip repetition code for a single logical qubit.
    Logical |0> = |000>, Logical |1> = |111>
    Stabilizer generators: Z_0 Z_1, Z_1 Z_2
    """
    def __init__(self):
        super().__init__(num_logical_qubits=1, num_physical_qubits=3)
        self.stabilizer_generators = [Pauli("ZZI"), Pauli("IZZ")] # Using Pauli objects consistent with return type
        self.distance = 3 # <--- CRITICAL FIX: ADD THIS LINE. Defines the distance for the repetition code.

    def get_encoding_circuit(self) -> QuantumCircuit:
        """
        Returns the Qiskit QuantumCircuit to encode |q> -> |q_L>.
        """
        qc = QuantumCircuit(self.num_physical_qubits, self.num_physical_qubits - self.num_logical_qubits) 
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

class SurfaceCode(QECCode):
    """
    Simplified Planar Distance-3 Surface Code (Surface-17).
    Encodes 1 logical qubit using 17 physical qubits (9 data, 8 ancilla).
    """
    def __init__(self, distance: int = 3):
        if distance != 3:
            print("Warning: Only distance-3 surface code (17 qubits) is fully detailed in Phase 2 for simplicity.")
        self.distance = distance # Define distance for SurfaceCode as well
        num_physical_qubits = 17 
        super().__init__(num_logical_qubits=1, num_physical_qubits=num_physical_qubits)

        # Stabilizer Generators for d=3 Surface Code (Pauli objects)
        # These are complex, for now, conceptual examples for a list.
        # You would typically have 4 Z-stabs and 4 X-stabs for d=3, 8 total.
        self.stabilizer_generators = [
            Pauli("ZZIIIIIIIIIZIIIIII"), 
            Pauli("IZIZIIIIIIIIZIIIII"), 
            Pauli("XIIIIXIIIIIIIIIIX"), 
            Pauli("IXIIIIXIIIIIIIIII"), 
        ]

    def get_encoding_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_physical_qubits)
        return qc 

    def get_syndrome_measurement_circuit(self) -> QuantumCircuit:
        num_classical_bits = 8 
        qc = QuantumCircuit(self.num_physical_qubits, num_classical_bits) 
        
        qc.cx(0, 9) 
        qc.cx(1, 9)
        qc.cx(3, 9)
        qc.cx(4, 9)
        qc.measure(9, 0) 

        qc.h(13) 
        qc.cx(13, 0) 
        qc.cx(13, 1)
        qc.cx(13, 3)
        qc.cx(13, 4)
        qc.h(13)
        qc.measure(13, 4) 
        
        self.syndrome_measurement_circuit = qc
        return qc