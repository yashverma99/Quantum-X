const API = "/api";
const state = { datasets: [], experiments: [], health: null, classical: null, quantum: null };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail || payload.message : payload;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

function notify(title, message, kind = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.innerHTML = `<strong>${escapeHtml(title)}</strong>${escapeHtml(message)}`;
  $("#toastRegion").append(toast);
  window.setTimeout(() => toast.remove(), 4800);
}

function setApiStatus(online, label) {
  const dot = $("#apiStatusDot");
  dot.classList.toggle("online", online);
  dot.classList.toggle("offline", !online);
  $("#apiStatusText").textContent = label;
}

function showPage(page) {
  const target = $(`#page-${page}`);
  if (!target) return;
  $$(".page").forEach((section) => section.classList.toggle("active", section === target));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === page));
  const label = $(`.nav-item[data-page="${page}"] span:nth-child(2)`)?.textContent || "Overview";
  $("#breadcrumbCurrent").textContent = label;
  document.title = `${label} · MediQAI`;
  $("#sidebar").classList.remove("open");
  if (page === "datasets") loadDatasets();
  if (page === "experiments") loadExperiments();
  if (page === "models") loadModels();
  history.replaceState(null, "", `#${page}`);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(date);
}

function updateCounts() {
  $("#statDatasets").textContent = state.datasets.length;
  $("#statExperiments").textContent = state.experiments.length;
  $("#navDatasetCount").textContent = state.datasets.length;
  $("#navExperimentCount").textContent = state.experiments.length;
  $("#lastUpdated").textContent = `Updated ${new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date())}`;
  renderRecentExperiments();
  renderExperiments();
  renderDatasets();
  updateDatasetOptions();
}

function renderRecentExperiments() {
  const host = $("#recentExperiments");
  const recent = state.experiments.slice(0, 5);
  if (!recent.length) {
    host.innerHTML = `<div class="empty-table">No experiments yet. Start a preprocessing pipeline to create your first run.</div>`;
    return;
  }
  host.innerHTML = `<table class="data-table"><thead><tr><th>EXPERIMENT</th><th>STATUS</th><th>FEATURES</th><th>CREATED</th></tr></thead><tbody>${recent.map((exp) => `<tr><td><strong>${escapeHtml(exp.name || exp.experiment_code)}</strong><br><span class="code-text">${escapeHtml(exp.experiment_code)}</span></td><td>${statusBadge(exp.status)}</td><td>${escapeHtml(exp.selected_features_count ?? "—")}</td><td>${formatDate(exp.created_at)}</td></tr>`).join("")}</tbody></table>`;
}

function statusBadge(status) {
  const normalized = String(status || "unknown").toLowerCase();
  const pending = normalized.includes("pending") || normalized.includes("await");
  return `<span class="badge ${pending ? "pending" : ""}">${escapeHtml(status || "Unknown")}</span>`;
}

function renderExperiments() {
  const host = $("#experimentsTable");
  const summary = $("#experimentSummary");
  if (!host) return;
  if (summary) summary.textContent = `${state.experiments.length} recorded ${state.experiments.length === 1 ? "run" : "runs"}`;
  if (!state.experiments.length) {
    host.innerHTML = `<div class="empty-table">No experiments recorded yet. Run preprocessing to create an experiment.</div>`;
    return;
  }
  host.innerHTML = `<table class="data-table"><thead><tr><th>EXPERIMENT</th><th>DATASET</th><th>STATUS</th><th>COMPONENTS</th><th>SEED</th><th>CREATED</th></tr></thead><tbody>${state.experiments.map((exp) => {
    const dataset = state.datasets.find((item) => item.id === exp.dataset_id);
    return `<tr><td><div class="experiment-name"><span class="experiment-dot"></span><span><strong>${escapeHtml(exp.name || "Preprocessing run")}</strong><br><span class="code-text">${escapeHtml(exp.experiment_code)}</span></span></div></td><td>${escapeHtml(dataset?.name || `Dataset #${exp.dataset_id ?? "—"}`)}</td><td>${statusBadge(exp.status)}</td><td>${escapeHtml(exp.selected_features_count ?? "—")}</td><td>${escapeHtml(exp.random_seed ?? "—")}</td><td>${formatDate(exp.created_at)}</td></tr>`;
  }).join("")}</tbody></table>`;
}

function renderDatasets() {
  const grid = $("#datasetGrid");
  if (!grid) return;
  const query = ($("#datasetSearch")?.value || "").trim().toLowerCase();
  const datasets = state.datasets.filter((dataset) => `${dataset.name} ${dataset.description || ""} ${dataset.target_column || ""}`.toLowerCase().includes(query));
  if (!datasets.length) {
    const title = query ? "No matching cohorts" : "No datasets yet";
    const message = query ? "Try another search term." : "Upload a CSV cohort or refresh to load the sample dataset.";
    grid.innerHTML = `<div class="empty-state"><div class="empty-icon">▤</div><strong>${title}</strong><p>${message}</p></div>`;
    return;
  }
  grid.innerHTML = datasets.map((dataset) => `<article class="dataset-card"><div class="dataset-card-top"><div class="dataset-title-group"><span class="dataset-icon">▤</span><div><h3>${escapeHtml(dataset.name)}</h3><p class="dataset-description">${escapeHtml(dataset.description || "Biomedical research cohort")}</p></div></div><span class="mini-status">Available</span></div><div class="dataset-metrics"><div class="dataset-metric"><strong>${Number(dataset.samples || 0).toLocaleString()}</strong><span>Samples</span></div><div class="dataset-metric"><strong>${Math.max(0, Number(dataset.features || 0) - 1).toLocaleString()}</strong><span>Features</span></div><div class="dataset-metric"><strong>${(Number(dataset.missing_ratio || 0) * 100).toFixed(1)}%</strong><span>Missing</span></div></div><div class="dataset-card-bottom"><span class="target-tag">Target · <strong>${escapeHtml(dataset.target_column || "not set")}</strong></span><div class="dataset-actions"><button class="small-action" data-preview="${Number(dataset.id)}">Preview</button><button class="small-action" data-use-dataset="${Number(dataset.id)}">Preprocess →</button></div></div></article>`).join("");
}

function updateDatasetOptions() {
  const select = $("#pipelineDataset");
  if (!select) return;
  const selected = select.value;
  if (!state.datasets.length) {
    select.innerHTML = `<option value="">No datasets available</option>`;
    return;
  }
  select.innerHTML = `<option value="">Choose a dataset…</option>${state.datasets.map((dataset) => `<option value="${Number(dataset.id)}">${escapeHtml(dataset.name)} (${Number(dataset.samples || 0).toLocaleString()} rows)</option>`).join("")}`;
  if (state.datasets.some((dataset) => String(dataset.id) === selected)) select.value = selected;
  else select.value = String(state.datasets[0].id);
  syncTargetColumn();
}

function syncTargetColumn() {
  const dataset = state.datasets.find((item) => String(item.id) === $("#pipelineDataset")?.value);
  if (dataset?.target_column) $("#targetColumn").value = dataset.target_column;
}

async function loadDatasets() {
  try {
    const response = await request("/datasets");
    state.datasets = Array.isArray(response.datasets) ? response.datasets : [];
    updateCounts();
  } catch (error) {
    $("#datasetGrid").innerHTML = `<div class="empty-state"><div class="empty-icon">!</div><strong>Could not load datasets</strong><p>${escapeHtml(error.message)}</p></div>`;
    notify("Dataset request failed", error.message, "error");
  }
}

async function loadExperiments() {
  try {
    const response = await request("/experiments");
    state.experiments = Array.isArray(response.experiments) ? response.experiments : [];
    updateCounts();
  } catch (error) {
    $("#experimentsTable").innerHTML = `<div class="empty-table">Could not load experiments: ${escapeHtml(error.message)}</div>`;
    notify("Experiment request failed", error.message, "error");
  }
}

async function loadModels() {
  $("#classicalModels").innerHTML = `<div class="loading-line"><span></span>Checking model availability</div>`;
  $("#quantumDetails").innerHTML = `<div class="loading-line"><span></span>Checking simulator</div>`;
  const [classicalResult, quantumResult] = await Promise.allSettled([request("/models/classical/status"), request("/quantum/status")]);
  if (classicalResult.status === "fulfilled") {
    state.classical = classicalResult.value;
    const models = state.classical.models || [];
    const anyAvailable = models.some((model) => model.available);
    $("#classicalStatusBadge").textContent = anyAvailable ? "Available" : "Unavailable";
    $("#classicalStatusBadge").classList.toggle("offline", !anyAvailable);
    $("#classicalModels").innerHTML = models.map((model) => `<div class="model-row"><span><strong>${escapeHtml(model.name)}</strong><br><small>${escapeHtml(model.type || "")}</small></span><span class="availability ${model.available ? "" : "no"}" title="${model.available ? "Available" : "Unavailable"}"></span></div>`).join("") || `<div class="empty-inline">No model status returned.</div>`;
  } else {
    $("#classicalStatusBadge").textContent = "Unavailable";
    $("#classicalStatusBadge").classList.add("offline");
    $("#classicalModels").innerHTML = `<div class="empty-inline">${escapeHtml(classicalResult.reason.message)}</div>`;
  }
  if (quantumResult.status === "fulfilled") {
    state.quantum = quantumResult.value;
    const available = state.quantum.quantum_simulator === "available";
    $("#quantumStatusBadge").textContent = available ? "Available" : "Unavailable";
    $("#quantumStatusBadge").classList.toggle("offline", !available);
    $("#quantumDetails").innerHTML = `<div class="quantum-detail-row"><span>Simulator</span><strong>${escapeHtml(state.quantum.quantum_simulator || "Unknown")}</strong></div><div class="quantum-detail-row"><span>Architectures</span><strong>${escapeHtml((state.quantum.supported_architectures || []).join(", ") || "—")}</strong></div><div class="quantum-detail-row"><span>Qubit range</span><strong>${escapeHtml((state.quantum.supported_qubits || []).join(" · ") || "—")}</strong></div><div class="quantum-detail-row"><span>Default shots</span><strong>${escapeHtml(state.quantum.default_shots || "—")}</strong></div>`;
  } else {
    $("#quantumStatusBadge").textContent = "Unavailable";
    $("#quantumStatusBadge").classList.add("offline");
    $("#quantumDetails").innerHTML = `<div class="empty-inline">${escapeHtml(quantumResult.reason.message)}</div>`;
  }
}

async function loadDashboard() {
  setApiStatus(false, "Checking API");
  try {
    const health = await request("/health");
    state.health = health;
    const healthy = health.status === "ok";
    setApiStatus(healthy, healthy ? "API online" : "API degraded");
    $("#statHealth").textContent = healthy ? "Online" : "Degraded";
    $("#statHealth").classList.toggle("online-text", healthy);
    $("#statHealth").classList.toggle("offline-text", !healthy);
    $("#healthCaption").textContent = `Python ${health.python || "—"}`;
    const quantumAvailable = health.quantum_simulator === "available";
    $("#statQuantum").textContent = quantumAvailable ? "Available" : "Unavailable";
    $("#quantumCaption").textContent = health.database === "configured" ? "Qiskit Aer · Database connected" : "Qiskit Aer runtime status";
    $("#quantumBadge").textContent = quantumAvailable ? "Ready" : "Offline";
    $("#quantumBadge").classList.toggle("offline", !quantumAvailable);
  } catch (error) {
    state.health = null;
    setApiStatus(false, "API offline");
    $("#statHealth").textContent = "Offline";
    $("#statHealth").classList.add("offline-text");
    $("#healthCaption").textContent = error.message || "Could not reach backend";
    $("#statQuantum").textContent = "Unknown";
    $("#quantumBadge").textContent = "—";
  }
  const results = await Promise.allSettled([request("/datasets"), request("/experiments")]);
  if (results[0].status === "fulfilled") state.datasets = results[0].value.datasets || [];
  if (results[1].status === "fulfilled") state.experiments = results[1].value.experiments || [];
  updateCounts();
}

function showPipelineResult(result) {
  const host = $("#pipelineResult");
  const variance = Math.max(0, Math.min(100, Number(result.cumulative_explained_variance || 0) * 100));
  host.innerHTML = `<div class="result-content"><h3>Pipeline complete</h3><div class="result-code">${escapeHtml(result.experiment_code || `EXP-${result.experiment_id}`)}</div><div class="result-metrics"><div class="result-metric"><strong>${Number(result.train_samples || 0).toLocaleString()}</strong><span>Training samples</span></div><div class="result-metric"><strong>${Number(result.test_samples || 0).toLocaleString()}</strong><span>Test samples</span></div><div class="result-metric"><strong>${escapeHtml(result.pca_components || "—")}</strong><span>PCA components</span></div><div class="result-metric"><strong>${variance.toFixed(1)}%</strong><span>Variance captured</span></div></div><div class="variance-bar" aria-label="${variance.toFixed(1)} percent variance captured"><span style="width:${variance}%"></span></div><div class="vector-box">${escapeHtml(JSON.stringify(result.sample_quantum_vector || []))}</div><p class="result-note">${escapeHtml(result.quantum_encoding || "Quantum-ready representation")} · Leakage-safe split confirmed.</p></div>`;
}

async function previewDataset(id) {
  const dataset = state.datasets.find((item) => item.id === id);
  $("#previewTitle").textContent = dataset?.name || "Cohort preview";
  $("#previewContent").innerHTML = `<div class="loading-line"><span></span>Loading sample rows</div>`;
  $("#previewDialog").showModal();
  try {
    const preview = await request(`/datasets/${id}/preview`);
    const columns = preview.columns || [];
    const rows = preview.rows || [];
    $("#previewContent").innerHTML = `<p class="preview-meta">Showing ${Number(preview.preview_count || rows.length)} of ${Number(preview.total_rows || 0).toLocaleString()} rows · ${Number(preview.total_cols || columns.length)} columns</p><div class="table-wrap preview-table"><table class="data-table"><thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}<br><span style="font-weight:400;color:#adb4bf">${escapeHtml(preview.dtypes?.[column] || "")}</span></th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(row[column] ?? "—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  } catch (error) {
    $("#previewContent").innerHTML = `<div class="empty-inline">Could not load this dataset preview: ${escapeHtml(error.message)}</div>`;
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const file = $("#datasetFile").files[0];
  if (!file) return notify("Choose a CSV", "Select a .csv file before uploading.", "error");
  if (file.size > 4 * 1024 * 1024) return notify("File is too large", "Vercel requests are limited to 4 MB for this upload.", "error");
  const button = $("#uploadButton");
  button.disabled = true;
  button.innerHTML = `<span class="loading-line"><span></span>Uploading…</span>`;
  try {
    const response = await request("/datasets/upload", { method: "POST", body: new FormData(form) });
    $("#uploadDialog").close();
    form.reset();
    $("#fileName").textContent = "Choose a CSV file";
    notify("Dataset uploaded", `${response.name || "Your cohort"} is ready in this workspace.`);
    await loadDashboard();
    showPage("datasets");
  } catch (error) {
    notify("Upload failed", error.message, "error");
  } finally {
    button.disabled = false;
    button.innerHTML = `Upload dataset <span>→</span>`;
  }
}

async function handlePipeline(event) {
  event.preventDefault();
  const datasetId = Number($("#pipelineDataset").value);
  if (!datasetId) return notify("Choose a dataset", "Select an available cohort to continue.", "error");
  const testSize = Number($("#testSize").value);
  if (testSize < 0.05 || testSize > 0.5) return notify("Invalid test split", "Enter a value from 0.05 to 0.50.", "error");
  const button = $("#runPipelineButton");
  button.disabled = true;
  button.innerHTML = `<span class="loading-line"><span></span>Running pipeline…</span>`;
  try {
    const result = await request("/preprocessing/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, target_column: $("#targetColumn").value.trim(), test_size: testSize, random_seed: Number($("#randomSeed").value), missing_strategy: "median", scaler: "standard", variance_threshold: 0, pca_components: Number($("#pcaComponents").value) })
    });
    showPipelineResult(result);
    notify("Preprocessing complete", `${result.experiment_code} · ${result.train_samples} training and ${result.test_samples} test samples.`);
    await Promise.all([loadExperiments(), loadDatasets()]);
  } catch (error) {
    notify("Pipeline failed", error.message, "error");
  } finally {
    button.disabled = false;
    button.innerHTML = `Run preprocessing <span>→</span>`;
  }
}

async function handlePreviewClick(event) {
  const button = event.target.closest("[data-preview]");
  if (button) await previewDataset(Number(button.dataset.preview));
  const useButton = event.target.closest("[data-use-dataset]");
  if (useButton) {
    const id = Number(useButton.dataset.useDataset);
    showPage("pipeline");
    $("#pipelineDataset").value = String(id);
    syncTargetColumn();
  }
}

function closeDialog(id) {
  const dialog = document.getElementById(id);
  if (dialog?.open) dialog.close();
}

function initialize() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => showPage(item.dataset.page)));
  $$('[data-go]').forEach((item) => item.addEventListener("click", () => showPage(item.dataset.go)));
  $("#menuToggle").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#refreshButton").addEventListener("click", loadDashboard);
  $("#reloadDatasets").addEventListener("click", loadDatasets);
  $("#reloadExperiments").addEventListener("click", loadExperiments);
  $("#refreshModels").addEventListener("click", loadModels);
  $("#datasetSearch").addEventListener("input", renderDatasets);
  $("#datasetGrid").addEventListener("click", handlePreviewClick);
  $("#pipelineDataset").addEventListener("change", syncTargetColumn);
  $("#pipelineForm").addEventListener("submit", handlePipeline);
  $("#uploadForm").addEventListener("submit", handleUpload);
  $("#openUpload").addEventListener("click", () => $("#uploadDialog").showModal());
  $("#datasetFile").addEventListener("change", (event) => { $("#fileName").textContent = event.target.files[0]?.name || "Choose a CSV file"; });
  $$('[data-close]').forEach((button) => button.addEventListener("click", () => closeDialog(button.dataset.close)));
  $$(".modal").forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); }));
  const requestedPage = location.hash.slice(1);
  showPage(["overview", "datasets", "pipeline", "experiments", "models"].includes(requestedPage) ? requestedPage : "overview");
  loadDashboard();
}

document.addEventListener("DOMContentLoaded", initialize);
