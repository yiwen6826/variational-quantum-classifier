import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append('../src')
from ansatz import build_vqc, get_prob

def cost(theta_vals, X_train, y_train, n_layers, shots=1000):
    '''
    Requires:
        theta_vals: array-like w/ len = n_qubits * n_layers (n_qubits = 4 for Iris)
            current parameter vals
        X_train: array-like
            training features (4)
        y_train: array-like
            binary classification labels (0 or 1)
        n_layers: int
            # of layers for ansatz
        shots: int; default 1000
            # of circuit runs
    Returns:
        float: avg binary cross-entropy loss over training set
    '''
    total_loss = 0.0
    for x, y in zip(X_train, y_train):
        qc, theta = build_vqc(x, n_layers=n_layers)
        p = get_prob(qc, theta, theta_vals, shots)

        p = np.clip(p, 1e-15, 1 - 1e-15) # avoid log(0)
        total_loss += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    return total_loss / len(X_train)

def train(X_train, y_train, n_layers=1, shots=500, maxiter=150, seed=42, verbose=True):
    '''
    Effect:
        Trains the VQC using COBYLA.

    Returns:
        theta_opt : array
            optimized parameters
        loss_history : list of float
            loss at each iteration
    '''
    np.random.seed(seed)
    theta_init = np.random.uniform(0, 2 * np.pi, 4 * n_layers)
    loss_history = []

    def track_costs(theta_vals):
        l = cost(theta_vals, X_train, y_train, n_layers=n_layers, shots=shots)
        loss_history.append(l)
        if verbose and len(loss_history) % 25 == 0:
            print(f"Iter {len(loss_history)}: loss = {l:.4f}")
        return l

    result = minimize(track_costs, theta_init, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
    return result.x, loss_history

def predict(x, theta_vals, n_layers=1, shots=1000):
    '''
    Returns:
        int(p > 0.5): prediction of the sample's label
            1 if P(|1>) > 0.5, 0 otherwise
        p: int; raw probability number for reference
    '''
    qc, theta = build_vqc(x, n_layers=n_layers)
    p = get_prob(qc, theta, theta_vals, shots)

    return int(p > 0.5), p

def evaluate(X_test, y_test, theta_vals, n_layers=1, shots=1000):
    '''
    Effect:
        evaluates VQC accuracy on test set
    Returns:
        accuracy: float
        y_preds: array of predicted labels
        y_probs: array of P(|1>) values
    '''
    y_preds = []
    y_probs = []
    for x in X_test:
        label, prob = predict(x, theta_vals, n_layers, shots)
        y_preds.append(label)
        y_probs.append(prob)
    y_preds = np.array(y_preds)
    y_probs = np.array(y_probs)
    accuracy = np.mean(y_preds == y_test)
    return accuracy, y_preds, y_probs
