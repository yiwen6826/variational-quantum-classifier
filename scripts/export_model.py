"""
Trains the OVR VQC (matching notebooks/multiclass.ipynb, N_LAYERS=2 --
the best-performing depth from the ablation study) and exports the
optimized parameters + feature scaling to JSON for the client-side
JS demo (docs/assets/model.json).
"""
import json
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.append("src")
from multiclass import ovr_train, ovr_evaluate

N_LAYERS = 2
SHOTS = 500
MAXITER = 150

iris = load_iris()
X_full = iris.data
y_full = iris.target

scaler = MinMaxScaler((0, np.pi))
X_scaled = scaler.fit_transform(X_full)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=42
)

ovr_thetas, _ = ovr_train(
    X_train, y_train, n_classes=3, n_layers=N_LAYERS,
    shots=SHOTS, maxiter=MAXITER, seed=42, verbose=True
)

accuracy, _, _ = ovr_evaluate(X_test, y_test, ovr_thetas, n_layers=N_LAYERS, shots=SHOTS)
print(f"\n3-class OVR test accuracy ({N_LAYERS} layers): {accuracy:.2%}")

output = {
    "n_layers": N_LAYERS,
    "n_qubits": 4,
    "feature_names": ["sepal length (cm)", "sepal width (cm)",
                       "petal length (cm)", "petal width (cm)"],
    "class_names": list(iris.target_names),
    "scaler_min": scaler.data_min_.tolist(),
    "scaler_max": scaler.data_max_.tolist(),
    "scaler_range": [0, float(np.pi)],
    "thetas": {str(c): ovr_thetas[c].tolist() for c in ovr_thetas},
    "test_accuracy": float(accuracy),
}

with open("docs/assets/model.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nWrote docs/assets/model.json")
