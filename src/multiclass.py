import numpy as np
from sklearn.metrics import accuracy_score
from ansatz import build_vqc, get_prob
from training import train, evaluate

def make_binary_labels(y, positive_class):
    '''
    Effect:
        convert multiple labels to binary OVR
        positive class = 1, everything else = 0
    Requires:
        y: array
        positive_class: int
    '''
    return (y == positive_class).astype(int)


def ovr_train(X_train, y_train, n_classes=3, n_layers=1,
              shots=500, maxiter=150, seed=42, verbose=True):
    '''
    Effects:
        Trains one-vs-rest VQCs for multiclass classification

    Requires:
        X_train: np.ndarray, shape (n_samples, 4)
        y_train: np.ndarray of integer class labels
        n_classes: int
        n_layers: int
        shots, maxiter, seed: training hyperparameters
        verbose: bool

    Returns:
        ovr_thetas: dict {class_label: np.ndarray of optimized params}
        ovr_loss_histories: dict {class_label: list of loss values}
    '''
    ovr_loss_history = {}
    ovr_thetas = {}

    for c in range(n_classes):
        if verbose:
            print(f"Training VQC {c} (class {c} vs rest)...")
        y_bin = make_binary_labels(y_train, positive_class=c)
        theta_opt_c, loss_history_c = train(
            X_train, y_bin,
            n_layers=n_layers, shots=shots,
            maxiter=maxiter, seed=seed + c,
            verbose=verbose
        )
        ovr_loss_history[c] = loss_history_c
        ovr_thetas[c] = theta_opt_c

    return ovr_thetas, ovr_loss_history


def ovr_predict(x, ovr_thetas, n_layers=1, shots=1000):
    '''
    Effect:
        predicts the class of x using one-vs-rest VQCs
        runs all 3 classifiers and returns the class with highest P(|1>)

    Requires:
        x: array-like, shape (4,)
            normalized feature vector
        ovr_thetas: dict {class_label: theta_vals}
            optimized parameters for each binary VQC
        n_layers: int
        shots: int

    Returns:
        predicted: int (0, 1, or 2)
        probs: dict {class_label: P(|1>)}
    '''
    probs = {}
    for c, theta_vals in ovr_thetas.items():
        qc, theta = build_vqc(x, n_layers=n_layers)
        probs[c] = get_prob(qc, theta, theta_vals, shots)
    predicted = max(probs, key=probs.get)

    return predicted, probs


def ovr_evaluate(X_test, y_test, ovr_thetas, n_layers=1, shots=1000):
    '''
    Effect:
        Evaluates 3-class OVR accuracy on a test set

    Returns:
        accuracy: float
        y_preds_ovr: np.ndarray of predicted labels
        y_probs_ovr: list of dicts
    '''
    y_preds_ovr = []
    y_probs_ovr = []
    for x in X_test:
        y_pred_ovr, y_prob_ovr = ovr_predict(x, ovr_thetas, n_layers=n_layers, shots=shots)
        y_preds_ovr.append(y_pred_ovr)
        y_probs_ovr.append(y_prob_ovr)
    accuracy = accuracy_score(y_test, np.array(y_preds_ovr))
    return accuracy, y_preds_ovr, y_probs_ovr