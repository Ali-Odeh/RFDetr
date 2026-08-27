const form = document.querySelector("#analyzeForm");
const imageInput = document.querySelector("#imageInput");
const labelInput = document.querySelector("#labelInput");
const dropZone = document.querySelector("#dropZone");
const selectedFile = document.querySelector("#selectedFile");
const selectedLabel = document.querySelector("#selectedLabel");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const progress = document.querySelector("#progress");
const errorMessage = document.querySelector("#errorMessage");
const analyzeButton = document.querySelector("#analyzeButton");
const results = document.querySelector("#results");
const modelStatus = document.querySelector("#modelStatus");
let latestResult = null;

threshold.addEventListener("input", () => { thresholdValue.textContent = threshold.value; });
imageInput.addEventListener("change", () => {
    selectedFile.textContent = imageInput.files[0] ? `${imageInput.files[0].name} — ${formatBytes(imageInput.files[0].size)}` : "";
});
labelInput.addEventListener("change", () => {
    selectedLabel.textContent = labelInput.files[0]
        ? `${labelInput.files[0].name} — ${formatBytes(labelInput.files[0].size)}`
        : "بدون label: سيتم عرض inference فقط بدون Accuracy أو IoU.";
});

["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
    event.preventDefault(); dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
    event.preventDefault(); dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", event => {
    if (event.dataTransfer.files.length) {
        imageInput.files = event.dataTransfer.files;
        imageInput.dispatchEvent(new Event("change"));
    }
});

form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!imageInput.files[0]) return;
    progress.classList.remove("hidden");
    errorMessage.classList.add("hidden");
    analyzeButton.disabled = true;
    results.classList.add("hidden");

    const data = new FormData();
    data.append("image", imageInput.files[0]);
    if (labelInput.files[0]) data.append("label", labelInput.files[0]);
    data.append("threshold", threshold.value);
    data.append("tile_size", document.querySelector("#tileSize").value);
    data.append("overlap", document.querySelector("#overlap").value);
    data.append("evaluation_iou", document.querySelector("#evaluationIou").value);

    try {
        const response = await fetch("/api/analyze", { method: "POST", body: data });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Analysis failed.");
        latestResult = payload;
        renderResult(payload);
    } catch (error) {
        errorMessage.textContent = error.message;
        errorMessage.classList.remove("hidden");
    } finally {
        progress.classList.add("hidden");
        analyzeButton.disabled = false;
    }
});

function renderResult(data) {
    document.querySelector("#totalCount").textContent = data.total.toLocaleString();
    setClassMetric("healthy", data, "healthy seed");
    setClassMetric("bad", data, "bad seed");
    setClassMetric("impurity", data, "impurity");
    document.querySelector("#qualityStatus").textContent = data.quality.label;
    document.querySelector("#resultImage").src = data.annotated_image;
    document.querySelector("#downloadImage").href = data.annotated_image;

    renderEvaluation(data.evaluation);

    const p = data.processing;
    const details = {
        "Image": `${data.image_width} × ${data.image_height}`,
        "Processing time": `${(p.elapsed_ms / 1000).toFixed(2)} sec`,
        "Tiles processed": p.tiles_processed,
        "Adaptive splits": p.saturated_tiles,
        "Raw predictions": p.raw_predictions,
        "Discarded parent predictions": p.adaptive_parent_predictions_discarded,
        "Duplicates removed": p.duplicates_removed,
        "Device": p.device.toUpperCase(),
        "Model": p.model_variant,
        "FP16 optimized": p.fp16_optimized ? "Yes" : "No",
        "Tile size": `${p.tile_size}px`,
        "Overlap": `${Math.round(p.overlap * 100)}%`,
    };
    document.querySelector("#processingDetails").innerHTML = Object.entries(details)
        .map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join("");

    document.querySelector("#tableCount").textContent = `${data.detections.length.toLocaleString()} instances`;
    document.querySelector("#detectionsBody").innerHTML = data.detections.map(item => `
        <tr><td>${item.id}</td><td><span class="class-badge">${escapeHtml(item.class_name)}</span></td>
        <td>${(item.confidence * 100).toFixed(1)}%</td><td>${item.center[0]}, ${item.center[1]}</td></tr>`).join("");
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderEvaluation(evaluation) {
    const section = document.querySelector("#evaluationSection");
    const countComparison = document.querySelector("#classCountComparison");
    const technicalEvaluation = document.querySelector("#technicalEvaluation");
    if (!evaluation) {
        section.classList.add("hidden");
        countComparison.classList.add("hidden");
        technicalEvaluation.classList.add("hidden");
        return;
    }
    section.classList.remove("hidden");
    countComparison.classList.remove("hidden");
    technicalEvaluation.classList.remove("hidden");
    document.querySelector("#evaluationThreshold").textContent = `Mask IoU ≥ ${evaluation.iou_threshold.toFixed(2)}`;
    document.querySelector("#gtTotal").textContent = evaluation.ground_truth_total.toLocaleString();
    document.querySelector("#meanIou").textContent = percent(evaluation.mean_mask_iou);
    document.querySelector("#evalPrecision").textContent = percent(evaluation.precision);
    document.querySelector("#evalRecall").textContent = percent(evaluation.recall);
    document.querySelector("#evalF1").textContent = percent(evaluation.f1);
    document.querySelector("#classAccuracy").textContent = percent(evaluation.accuracy ?? evaluation.matched_class_accuracy);
    document.querySelector("#evaluationCounts").innerHTML = [
        ["TP", evaluation.true_positive], ["FP", evaluation.false_positive],
        ["FN", evaluation.false_negative], ["Spatial matches", evaluation.spatial_matches]
    ].map(([name, value]) => `<span><b>${name}</b> ${Number(value).toLocaleString()}</span>`).join("");

    const comparisonRows = evaluation.per_class.map(item => ({
        className: item.class_name,
        groundTruth: item.ground_truth,
        prediction: item.predictions,
        difference: item.predictions - item.ground_truth,
    }));
    document.querySelector("#classCountBody").innerHTML = comparisonRows.map(item => `
        <tr><td><span class="class-badge">${escapeHtml(item.className)}</span></td>
        <td>${item.groundTruth.toLocaleString()}</td><td>${item.prediction.toLocaleString()}</td>
        <td class="${differenceClass(item.difference)}">${formatDifference(item.difference)}</td></tr>`).join("");
    const totalDifference = evaluation.predictions_total - evaluation.ground_truth_total;
    document.querySelector("#classCountFoot").innerHTML = `
        <tr><th>Total</th><th>${evaluation.ground_truth_total.toLocaleString()}</th>
        <th>${evaluation.predictions_total.toLocaleString()}</th>
        <th class="${differenceClass(totalDifference)}">${formatDifference(totalDifference)}</th></tr>`;

    document.querySelector("#evaluationBody").innerHTML = evaluation.per_class.map(item => `
        <tr><td>${escapeHtml(item.class_name)}</td><td>${item.ground_truth}</td><td>${item.predictions}</td>
        <td>${item.true_positive}</td><td>${percent(item.precision)}</td><td>${percent(item.recall)}</td>
        <td>${percent(item.f1)}</td><td>${percent(item.mean_iou)}</td></tr>`).join("");

    const matrix = evaluation.confusion_matrix;
    document.querySelector("#confusionTable").innerHTML = `
        <thead><tr><th>GT \\ Pred</th>${matrix.labels.map(label => `<th>${escapeHtml(label)}</th>`).join("")}</tr></thead>
        <tbody>${matrix.values.map((row, index) => `<tr><th>${escapeHtml(matrix.labels[index])}</th>${row.map(value => `<td>${value}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function formatDifference(value) { return value > 0 ? `+${value}` : String(value); }
function differenceClass(value) { return value === 0 ? "difference-zero" : "difference-nonzero"; }

function setClassMetric(prefix, data, className) {
    document.querySelector(`#${prefix}Count`).textContent = data.counts[className].toLocaleString();
    document.querySelector(`#${prefix}Percent`).textContent = `${data.percentages[className].toFixed(2)}%`;
}

document.querySelector("#downloadCsv").addEventListener("click", () => {
    if (!latestResult) return;
    const rows = [["id", "class_id", "class", "raw_class_id", "raw_class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]];
    latestResult.detections.forEach(item => rows.push([
        item.id, item.class_id, item.class_name, item.raw_class_id, item.raw_class_name,
        item.confidence, ...item.bbox, ...item.center
    ]));
    const csv = rows.map(row => row.map(csvValue).join(",")).join("\n");
    downloadBlob(csv, "multispector-detections.csv", "text/csv;charset=utf-8");
});

document.querySelector("#downloadJson").addEventListener("click", () => {
    if (!latestResult) return;
    const clean = { ...latestResult };
    delete clean.annotated_image;
    downloadBlob(JSON.stringify(clean, null, 2), "multispector-result.json", "application/json");
});

function downloadBlob(content, filename, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = filename; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
}
function csvValue(value) { const text = String(value); return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; }
function formatBytes(bytes) { return bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`; }
function percent(value) { return `${(Number(value) * 100).toFixed(2)}%`; }
function escapeHtml(value) { return value.replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]); }

fetch("/api/health").then(response => response.json()).then(data => {
    const ready = data.status === "ready";
    modelStatus.classList.add(ready ? "ready" : "error");
    const labels = {
        "ready": " Checkpoint ready",
        "missing-checkpoint": " Checkpoint missing",
        "corrupted-checkpoint": " Checkpoint download incomplete",
        "unreadable-checkpoint": " Checkpoint unreadable",
    };
    modelStatus.lastChild.textContent = ready
        ? ` ${data.model_variant} ready`
        : (labels[data.status] || " Checkpoint unavailable");
}).catch(() => {
    modelStatus.classList.add("error");
    modelStatus.lastChild.textContent = " Server unavailable";
});
