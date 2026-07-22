# Variational Quantum Classifier

A 4-qubit Variational Quantum Classifier (VQC), built with Qiskit and trained with the
COBYLA optimizer, that classifies flowers in the Iris dataset from their four
measurements (sepal length/width, petal length/width).

**[Try a live demo](https://yiwen6826.github.io/variational-quantum-classifier/)** —
move sliders for the four measurements and watch the model predict the species in your
browser.

## How it works

1. **Encode.** Each of the 4 measurements is scaled to an angle between 0 and π, then applied an `Ry` rotation 
    (one qubit per feature as mentioned). This transforms the classical Iris data into quantum states that 
    can be manipulated in the VQC. ([src/encoding.py](src/encoding.py)).
2. **Entangle.** A trained ansatz layer rotates the qubits, then applies a CNOT "staircase" that lets the qubits interact. 
    This is repeated once for a depth-2 VQC, which surfaced during ablation tests as the optimal circuit depth that maximizes 
    accuracy vs. parameter noise. ([src/ansatz.py](src/ansatz.py)).
3. **Measure.** Now that all qubit information is entangled, qubit 0 is measured. An OVR (one-vs-rest) approach is used, 
    leading to three separate circuits — one per species. Each circuit is evaluated, and the one most likely to read out |1⟩ wins. 
    ([src/multiclass.py](src/multiclass.py)).

Parameters are optimized with `scipy.optimize.minimize` using **COBYLA**, a
gradient-free optimizer, against a binary cross-entropy loss
([src/training.py](src/training.py)).

## Results

- Best-performing depth: **2 ansatz layers**, ~90-97% 3-class test accuracy (varies run to
  run due to shot noise in the simulator).
- A circuit-depth ablation (1/2/3 layers) shows accuracy peaks at 2 layers and slightly
  degrades at 3, likely from added gate noise ([results/depth_ablation.png](results/depth_ablation.png)).
- See [results/](results) for confusion matrices, loss curves, and circuit diagrams from
  training.

## Repo structure

```
src/            Core library: encoding, ansatz, training, OVR multiclass logic
notebooks/      Exploratory + training notebooks (ansatz design, encoding, full OVR training)
results/        Saved plots and figures from training runs
docs/           Static site (GitHub Pages) — browser demo of the trained OVR model
scripts/        Utility scripts, e.g. exporting trained parameters for the web demo
```

## Setup

Requires **Python 3.11+** (numpy's pinned version won't install on older Pythons).

```bash
pip install -r requirements.txt
```

Notebooks are meant to be run from the `notebooks/` directory (they add `../src` to the
path). Requires Qiskit + Qiskit Aer for circuit simulation.

## License

MIT — see [LICENSE](LICENSE).
