import numpy as np
from qiskit import QuantumCircuit

def encode_data(x):
    '''
    Requires:
        x: array-like w/ length 4
    Effect:
        Angle encodes normalized feature vector onto qubits using Ry rotation
    Returns: 
        qc: QuantumCircuit
            unmeasured quantum circuit
    '''
    qc = QuantumCircuit(len(x))
    for i, x_i in enumerate(x):
        qc.ry(x_i, i)
    return qc