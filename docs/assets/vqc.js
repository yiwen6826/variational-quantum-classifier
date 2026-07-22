/**
 * Client-side re-implementation of the trained OVR VQC (see src/ansatz.py).
 * The circuit has only Ry rotations and CNOTs, so the statevector stays
 * real-valued -- an exact (noiseless) statevector simulation reproduces
 * the same P(|1>) that get_prob() estimates from shots in Python.
 */

const N_QUBITS = 4;
const DIM = 1 << N_QUBITS; // 16

function zeroState() {
  const state = new Float64Array(DIM);
  state[0] = 1.0;
  return state;
}

// Apply Ry(theta) on `qubit` (0 = least-significant bit, matching Qiskit).
function applyRy(state, qubit, theta) {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  const mask = 1 << qubit;
  const out = state.slice();
  for (let i = 0; i < DIM; i++) {
    if ((i & mask) === 0) {
      const j = i | mask;
      const a0 = state[i];
      const a1 = state[j];
      out[i] = c * a0 - s * a1;
      out[j] = s * a0 + c * a1;
    }
  }
  return out;
}

// Apply CNOT(control, target).
function applyCx(state, control, target) {
  const cMask = 1 << control;
  const tMask = 1 << target;
  const out = state.slice();
  for (let i = 0; i < DIM; i++) {
    if ((i & cMask) !== 0) {
      out[i ^ tMask] = state[i];
    }
  }
  return out;
}

// Builds the full VQC (encoding + ansatz) and returns P(qubit0 == 1).
function getProb(x, thetaVals, nLayers) {
  let state = zeroState();

  // Angle-encode features: Ry(x_i) on qubit i
  for (let i = 0; i < N_QUBITS; i++) {
    state = applyRy(state, i, x[i]);
  }

  // Ansatz: n_layers of (Ry rotation layer + CNOT staircase)
  let idx = 0;
  for (let layer = 0; layer < nLayers; layer++) {
    for (let q = 0; q < N_QUBITS; q++) {
      state = applyRy(state, q, thetaVals[idx]);
      idx++;
    }
    for (let q = N_QUBITS - 1; q >= 1; q--) {
      state = applyCx(state, q, q - 1);
    }
  }

  // P(qubit 0 == 1) = sum of |amplitude|^2 over basis states with bit0 set
  let p1 = 0;
  for (let i = 0; i < DIM; i++) {
    if ((i & 1) !== 0) p1 += state[i] * state[i];
  }
  return p1;
}

// Scales a raw feature vector (in the dataset's native units) to [0, pi]
// using the same per-feature min/max the model was trained with.
function scaleFeatures(rawX, model) {
  const [lo, hi] = model.scaler_range;
  return rawX.map((v, i) => {
    const min = model.scaler_min[i];
    const max = model.scaler_max[i];
    const t = (v - min) / (max - min);
    return lo + t * (hi - lo);
  });
}

// Runs all 3 one-vs-rest classifiers and returns { predicted, probs }.
function ovrPredict(rawX, model) {
  const xScaled = scaleFeatures(rawX, model);
  const probs = {};
  for (const c of Object.keys(model.thetas)) {
    probs[c] = getProb(xScaled, model.thetas[c], model.n_layers);
  }
  let predicted = null;
  let best = -Infinity;
  for (const c of Object.keys(probs)) {
    if (probs[c] > best) {
      best = probs[c];
      predicted = c;
    }
  }
  return { predicted: Number(predicted), probs };
}
