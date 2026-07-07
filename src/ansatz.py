from encoding import encode_data
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
sim = AerSimulator()

def build_ansatz(n_qubits=4, n_layers=1):
    '''
    Parameters: 
        n_qubits: int
            number of qubits (default 4 Iris dataset features)
        n_layers: int
            number of layers (default 1)

    Returns: 
        qc: QuantumCircuit
            unmeasured quantum circuit
        theta: ParameterVector
            parameter vector w/ length n_qubits * n_layers
    '''
    qc = QuantumCircuit(n_qubits)
    theta = ParameterVector('theta', n_qubits * n_layers)

    idx = 0
    for _ in range(n_layers):
        # Rotation layer
        # apply Ry gate to each qubit as trainable weights
        for qubit in range(n_qubits):
            qc.ry(theta[idx], qubit)
            idx+=1
        
        # Entanglement layer
        # CNOT staircase ending at qubit 0, the measured qubit
        # creates correlation between all qubits
        for qubit in reversed(range(1, n_qubits)):
            qc.cx(qubit, qubit-1)
    
        qc.barrier()

    return qc, theta

def build_vqc(x, n_layers=1):
    '''
    Parameters:
        x: array-like w/ length 4 (in the case of the Iris dataset)
            length = number of qubits
        n_layers: int
            number of ansatz layers, default 1

    Returns:
        circuit: QuantumCircuit
            full vqc w/ encoding, ansatz, and measurement on qubit 0
        theta: ParameterVector
            trainable parameters w/ length len(x) * n_layers
    '''
    qc_enc = encode_data(x)
    ansatz, theta = build_ansatz(len(x), n_layers)

    circuit = QuantumCircuit(len(x), 1)

    circuit.compose(qc_enc, inplace=True)
    circuit.compose(ansatz, inplace=True)
    circuit.measure(0, 0)

    return circuit, theta

def get_prob(qc, theta, theta_vals, shots=1000):
    '''
    Effect:
        Finds the probability of a parameterized VQC returning |1>
    Parameters:
        qc: QuantumCircuit
            measured quantum circuit built by build_vqc
        theta: ParameterVector
            parameters used in qc
        theta_vals: array-like w/ length = len(theta)
            values to bind to the parameters
        shots: int
            number of runs; default 1000
    '''
    param_dict = dict(zip(theta, theta_vals))
    bound_qc = qc.assign_parameters(param_dict)

    job = sim.run(bound_qc, shots=shots)
    counts = job.result().get_counts()

    return counts.get('1', 0) / shots