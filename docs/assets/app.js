const COLORS = { 0: "var(--setosa)", 1: "var(--versicolor)", 2: "var(--virginica)" };

const PRESETS = {
  setosa: [5.1, 3.5, 1.4, 0.2],
  versicolor: [5.9, 3.0, 4.2, 1.5],
  virginica: [6.5, 3.0, 5.2, 2.0],
};

let model = null;
let currentX = null;

function buildSliders() {
  const container = document.getElementById("sliders");
  container.innerHTML = "";

  model.feature_names.forEach((name, i) => {
    const min = model.scaler_min[i];
    const max = model.scaler_max[i];
    const value = currentX[i];

    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <label>
        <span>${name}</span>
        <span class="value" id="value-${i}">${value.toFixed(1)} cm</span>
      </label>
      <input type="range" id="slider-${i}" min="${min}" max="${max}" step="0.1" value="${value}">
    `;
    container.appendChild(row);

    row.querySelector("input").addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      currentX[i] = v;
      document.getElementById(`value-${i}`).textContent = `${v.toFixed(1)} cm`;
      update();
    });
  });
}

function update() {
  const { predicted, probs } = ovrPredict(currentX, model);
  const speciesName = model.class_names[predicted];

  document.getElementById("predicted-name").textContent = speciesName;
  document.getElementById("prediction-card").className = `prediction-card ${speciesName}`;

  const barsEl = document.getElementById("prob-bars");
  barsEl.innerHTML = "";
  Object.keys(probs)
    .sort((a, b) => a - b)
    .forEach((c) => {
      const p = probs[c];
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-label">
          <span>${model.class_names[c]}</span>
          <span>P(|1&#10217;) = ${p.toFixed(3)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${(p * 100).toFixed(1)}%; background:${COLORS[c]}"></div>
        </div>
      `;
      barsEl.appendChild(row);
    });
}

function applyPreset(name) {
  currentX = [...PRESETS[name]];
  buildSliders();
  update();
}

async function init() {
  model = await fetch("assets/model.json", { cache: "no-store" }).then((r) => r.json());

  document.getElementById("accuracy-note").textContent =
    `Trained with COBYLA, ${model.n_layers} ansatz layers, ` +
    `${(model.test_accuracy * 100).toFixed(0)}% 3-class accuracy.`;

  currentX = [...PRESETS.setosa];
  buildSliders();
  update();

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });
}

init();
