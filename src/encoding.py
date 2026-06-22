import numpy as np
from qiskit import QuantumCircuit

def encode_data(x):
    qc = QuantumCircuit(len(x))
    for i, x_i in enumerate(x):
        qc.ry(x_i, i)
    return qc