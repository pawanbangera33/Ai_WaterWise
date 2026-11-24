/**
 * AI WaterWise - Single Unified Real-Time Monitoring Interface Controller
 * Enforces pure backend data lineage, live SSE streaming / polling, dynamic Chart.js rendering,
 * and immediate scenario switching.
 */

const API_BASE = (window.location.protocol === "file:") ? "http://127.0.0.1:8000/api" : "/api";

let isSimRunning = false;
let simTimer = null;
let currentSpeed = 1.0;
let activeCondition = "normal";

let liveChartInstance = null;
let forecastChartInstance = null;

let liveChartBuffer = {
    labels: [],
    actual: [],
    predicted: []
};
const MAX_LIVE_BUFFER = 60;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    fetchDatasetStatus();
    initCharts();
});

/**
 * Updates selected filename text label when user picks a CSV file
 */
function updateSelectedFilename(input) {
    const filenameSpan = document.getElementById("selected-filename");
    if (input.files && input.files[0]) {
        filenameSpan.innerText = input.files[0].name;
    } else {
        filenameSpan.innerText = "No file chosen";
    }
}

/**
 * Checks dataset and model status from GET /api/dataset-status
 */
async function fetchDatasetStatus() {
    try {
        const response = await fetch(`${API_BASE}/dataset-status`);
        if (!response.ok) return;
        const data = await response.json();
        
        const badgeProc = document.getElementById("badge-processed");
        const badgeTrain = document.getElementById("badge-trained");
        
        if (data.has_active_dataset && data.dataset_info) {
            const info = data.dataset_info;
            document.getElementById("info-filename").innerText = info.filename || "water_data.csv";
            document.getElementById("info-records").innerText = info.records || "0";
            
            if (info.is_processed) {
                badgeProc.className = "badge badge-success";
                badgeProc.innerText = "DATA PROCESSED";
            } else {
                badgeProc.className = "badge badge-warning";
                badgeProc.innerText = "UNPROCESSED";
            }
            
            if (data.models_valid_for_current || info.is_trained) {
                badgeTrain.className = "badge badge-success";
                badgeTrain.innerText = "MODELS TRAINED";
            } else {
                badgeTrain.className = "badge badge-warning";
                badgeTrain.innerText = "UNTRAINED";
            }
        }
    } catch (err) {
        console.error("Error fetching dataset status:", err);
    }
}

/**
 * Handles CSV File Upload (POST /api/upload)
 */
async function handleFileUpload(event) {
    if (event) event.preventDefault();
    const fileInput = document.getElementById("csv-file-input");
    const errBox = document.getElementById("upload-error-msg");
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please choose a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            errBox.innerText = `Upload Error: ${result.detail || "Failed to upload file."}`;
            errBox.classList.remove("hidden");
            return;
        }

        errBox.classList.add("hidden");
        document.getElementById("info-filename").innerText = result.filename;
        document.getElementById("info-records").innerText = result.records;
        
        fetchDatasetStatus();
    } catch (err) {
        console.error("Upload failed:", err);
    }
}

/**
 * Handles Dataset Processing (POST /api/process)
 */
async function handleProcessData() {
    const errBox = document.getElementById("upload-error-msg");
    const btnProcess = document.getElementById("btn-process");
    btnProcess.innerText = "Processing...";
    btnProcess.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/process`, { method: 'POST' });
        const result = await response.json();

        btnProcess.innerText = "Process Data";
        btnProcess.disabled = false;

        if (!response.ok) {
            errBox.innerText = `Processing Error: ${result.detail}`;
            errBox.classList.remove("hidden");
            return;
        }

        errBox.classList.add("hidden");
        fetchDatasetStatus();
    } catch (err) {
        console.error("Processing error:", err);
        btnProcess.innerText = "Process Data";
        btnProcess.disabled = false;
    }
}

/**
 * Handles Model Training (POST /api/train)
 */
async function handleTrainModels() {
    const errBox = document.getElementById("upload-error-msg");
    const btnTrain = document.getElementById("btn-train");
    btnTrain.innerText = "Training Models...";
    btnTrain.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/train`, { method: 'POST' });
        const result = await response.json();

        btnTrain.innerText = "Train Models";
        btnTrain.disabled = false;

        if (!response.ok) {
            errBox.innerText = `Training Error: ${result.detail}`;
            errBox.classList.remove("hidden");
            return;
        }

        errBox.classList.add("hidden");
        fetchDatasetStatus();
        updateModelDetails(result.metrics);
        addEventLog("SYSTEM", "All models successfully trained on active dataset.", "info");
    } catch (err) {
        console.error("Training error:", err);
        btnTrain.innerText = "Train Models";
        btnTrain.disabled = false;
    }
}

/* ==========================================================================
   REAL-TIME STREAMING & TELEMETRY CONTROLLER
   ========================================================================== */

/**
 * Toggles Real-time Simulation start/pause
 */
async function toggleRealtimeSim() {
    const btnToggle = document.getElementById("btn-toggle-sim");
    
    if (!isSimRunning) {
        try {
            const res = await fetch(`${API_BASE}/simulation/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ condition: activeCondition, speed: currentSpeed })
            });
            if (!res.ok) throw new Error("Failed to start simulation.");
            
            isSimRunning = true;
            btnToggle.innerText = "PAUSE LIVE MONITORING";
            btnToggle.className = "btn btn-danger btn-lg";
            
            updateStatusIndicator("RUNNING", "dot-running");
            addEventLog("SYSTEM", `Live telemetry streaming STARTED in scenario: ${activeCondition.toUpperCase()}`, "info");
            
            startSimLoop();
        } catch (err) {
            console.error("Start sim error:", err);
            alert("Could not start live simulation feed.");
        }
    } else {
        try {
            await fetch(`${API_BASE}/simulation/pause`, { method: "POST" });
            isSimRunning = false;
            btnToggle.innerText = "START LIVE MONITORING";
            btnToggle.className = "btn btn-primary btn-lg";
            
            updateStatusIndicator("PAUSED", "dot-paused");
            addEventLog("SYSTEM", "Live telemetry streaming PAUSED.", "warning");
            stopSimLoop();
        } catch (err) {
            console.error("Pause sim error:", err);
        }
    }
}

function startSimLoop() {
    stopSimLoop();
    fetchSimTick();
    simTimer = setInterval(fetchSimTick, currentSpeed * 1000);
}

function stopSimLoop() {
    if (simTimer) {
        clearInterval(simTimer);
        simTimer = null;
    }
}

async function stepRealtimeSim() {
    await fetchSimTick();
}

async function stopRealtimeSim() {
    try {
        await fetch(`${API_BASE}/simulation/stop`, { method: "POST" });
    } catch (e) {}
    
    isSimRunning = false;
    stopSimLoop();
    
    const btnToggle = document.getElementById("btn-toggle-sim");
    btnToggle.innerText = "START LIVE MONITORING";
    btnToggle.className = "btn btn-primary btn-lg";
    
    updateStatusIndicator("STOPPED", "dot-stopped");
    resetTelemetryUI();
    addEventLog("SYSTEM", "Live monitoring stopped and state reset.", "info");
}

async function selectSimCondition(cond) {
    activeCondition = cond;
    
    // Visually highlight selected scenario button
    document.querySelectorAll(".scenario-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`scen-btn-${cond}`);
    if (activeBtn) activeBtn.classList.add("active");
    
    try {
        await fetch(`${API_BASE}/simulation/condition`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ condition: cond })
        });
        
        addEventLog("SCENARIO", `Operational scenario switched to: ${cond.toUpperCase()}`, "warning");
    } catch (err) {
        console.error("Condition change error:", err);
    }
    
    if (!isSimRunning) {
        fetchSimTick();
    }
}

async function changeRealtimeSpeed(val) {
    currentSpeed = parseFloat(val);
    if (isSimRunning) {
        startSimLoop();
    }
}

/**
 * Fetches and processes next telemetry tick from GET /api/simulation/tick
 */
async function fetchSimTick() {
    try {
        const res = await fetch(`${API_BASE}/simulation/tick`);
        if (!res.ok) return;
        const data = await res.json();
        const tick = data.tick;
        if (!tick) return;
        
        updateTelemetryUI(tick);
    } catch (err) {
        console.error("Error fetching telemetry tick:", err);
    }
}

/**
 * Updates DOM elements with telemetry object strictly returned from backend
 */
function updateTelemetryUI(tick) {
    // 5. CURRENT LIVE TELEMETRY
    document.getElementById("val-consumption").innerText = `${tick.water_consumption} L/hr`;
    document.getElementById("val-temp").innerText = `${tick.temperature} °C`;
    document.getElementById("val-humidity").innerText = `${tick.humidity} %`;
    document.getElementById("val-occupancy").innerText = tick.occupancy;
    document.getElementById("val-tank").innerText = `${tick.tank_level} %`;
    document.getElementById("val-timestamp").innerText = tick.timestamp;

    // 6. AI DEMAND PREDICTION
    document.getElementById("val-actual").innerText = `${tick.water_consumption} L/hr`;
    document.getElementById("val-dnn-pred").innerText = tick.predicted_consumption !== null ? `${tick.predicted_consumption} L/hr` : "--";
    
    const diffElem = document.getElementById("val-dnn-diff");
    const diffVal = tick.prediction_difference !== undefined ? tick.prediction_difference : (tick.water_consumption - (tick.predicted_consumption || 0));
    diffElem.innerText = `${diffVal > 0 ? '+' : ''}${roundVal(diffVal, 1)} L/hr`;
    diffElem.style.color = Math.abs(diffVal) > 200 ? "#dc2626" : "#2563eb";

    // 8. ANOMALY DETECTION
    const aeBadge = document.getElementById("val-anomaly-status");
    aeBadge.innerText = tick.anomaly_status;
    if (tick.anomaly_status === "ANOMALY DETECTED" || tick.is_anomaly) {
        aeBadge.className = "status-badge badge-anomaly-status";
    } else {
        aeBadge.className = "status-badge badge-normal-status";
    }
    
    document.getElementById("val-ae-error").innerText = tick.reconstruction_error;
    document.getElementById("val-ae-thresh").innerText = tick.anomaly_threshold;
    document.getElementById("val-anomaly-type").innerText = tick.anomaly_type || tick.condition_label || "NORMAL";

    // 9. LEAKAGE RISK
    const riskElem = document.getElementById("val-risk-prob");
    riskElem.innerText = `${tick.leakage_percentage}%`;
    if (tick.leakage_probability > 0.6) {
        riskElem.style.color = "#dc2626";
    } else if (tick.leakage_probability > 0.3) {
        riskElem.style.color = "#d97706";
    } else {
        riskElem.style.color = "#16a34a";
    }

    const evidenceList = document.getElementById("val-risk-evidence");
    if (tick.leakage_evidence && Array.isArray(tick.leakage_evidence)) {
        evidenceList.innerHTML = tick.leakage_evidence.map(item => `<li>${item}</li>`).join("");
    }
    document.getElementById("val-risk-exp").innerText = tick.leakage_explanation;

    // 10. Q-LEARNING DECISION
    const actionBadge = document.getElementById("val-q-action");
    actionBadge.innerText = tick.recommended_action;
    if (tick.recommended_action.includes("SHUTOFF") || tick.recommended_action.includes("INVESTIGATE") || tick.recommended_action.includes("EMERGENCY")) {
        actionBadge.className = "q-action-badge action-danger";
    } else {
        actionBadge.className = "q-action-badge action-normal";
    }

    if (tick.q_state) {
        document.getElementById("val-q-tank").innerText = tick.q_state.tank || "--";
        document.getElementById("val-q-demand").innerText = tick.q_state.demand || "--";
        document.getElementById("val-q-risk").innerText = tick.q_state.risk || "--";
        document.getElementById("val-q-anom").innerText = tick.q_state.anomaly || "--";
    }
    document.getElementById("val-q-reason").innerText = tick.recommendation_reason || tick.q_reason;

    // 7. FUTURE DEMAND FORECAST CHART (LSTM 24h)
    if (tick.lstm_forecast && Array.isArray(tick.lstm_forecast)) {
        updateForecastChart(tick.lstm_forecast);
    }

    // 11. LIVE TELEMETRY CHART
    appendLiveChartPoint(tick.timestamp, tick.water_consumption, tick.predicted_consumption);

    // 12. LIVE EVENT LOG
    addEventLog(
        tick.anomaly_status,
        `Consumption: ${tick.water_consumption} L/hr | DNN: ${tick.predicted_consumption || '--'} L/hr | Risk: ${tick.leakage_percentage}% | Scenario: ${tick.anomaly_type}`,
        tick.anomaly_status === "ANOMALY DETECTED" ? "anomaly" : "normal"
    );
}

function resetTelemetryUI() {
    document.getElementById("val-consumption").innerText = "-- L/hr";
    document.getElementById("val-temp").innerText = "-- °C";
    document.getElementById("val-humidity").innerText = "-- %";
    document.getElementById("val-occupancy").innerText = "--";
    document.getElementById("val-tank").innerText = "-- %";
    document.getElementById("val-timestamp").innerText = "--:--:--";
    
    document.getElementById("val-actual").innerText = "-- L/hr";
    document.getElementById("val-dnn-pred").innerText = "-- L/hr";
    document.getElementById("val-dnn-diff").innerText = "-- L/hr";
    
    const aeBadge = document.getElementById("val-anomaly-status");
    aeBadge.innerText = "NORMAL";
    aeBadge.className = "status-badge badge-normal-status";
    
    document.getElementById("val-ae-error").innerText = "0.0000";
    document.getElementById("val-anomaly-type").innerText = "NORMAL";
    
    document.getElementById("val-risk-prob").innerText = "0.0%";
    document.getElementById("val-risk-prob").style.color = "#16a34a";
    document.getElementById("val-risk-evidence").innerHTML = "<li>Awaiting telemetry stream...</li>";
    document.getElementById("val-risk-exp").innerText = "System operating within normal parameters.";
    
    const actionBadge = document.getElementById("val-q-action");
    actionBadge.innerText = "PUMP OFF";
    actionBadge.className = "q-action-badge action-normal";
    document.getElementById("val-q-reason").innerText = "Real-time telemetry consistent with normal equilibrium bounds.";
    
    liveChartBuffer = { labels: [], actual: [], predicted: [] };
    initCharts();
}

function updateStatusIndicator(text, cssClass) {
    document.getElementById("status-text").innerText = text;
    const dot = document.getElementById("status-dot");
    dot.className = `status-dot ${cssClass}`;
}

/* ==========================================================================
   CHARTS & EVENT LOG CONTROLLERS
   ========================================================================== */

function initCharts() {
    // 11. Live Water Consumption Chart
    const ctxLive = document.getElementById("liveChart").getContext("2d");
    if (liveChartInstance) liveChartInstance.destroy();
    
    liveChartInstance = new Chart(ctxLive, {
        type: 'line',
        data: {
            labels: liveChartBuffer.labels,
            datasets: [
                {
                    label: 'Actual Consumption (L/hr)',
                    data: liveChartBuffer.actual,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 2.5,
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'DNN Expected Consumption (L/hr)',
                    data: liveChartBuffer.predicted,
                    borderColor: '#16a34a',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.2,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: { title: { display: true, text: 'Time / Telemetry Tick' } },
                y: { title: { display: true, text: 'Water Consumption (L/hr)' } }
            }
        }
    });

    // 7. Future Demand Forecast Chart (LSTM 24h)
    const ctxForecast = document.getElementById("forecastChart").getContext("2d");
    if (forecastChartInstance) forecastChartInstance.destroy();

    forecastChartInstance = new Chart(ctxForecast, {
        type: 'line',
        data: {
            labels: Array.from({length: 24}, (_, i) => `+${i+1}h`),
            datasets: [{
                label: 'LSTM Predicted Future Demand (L/hr)',
                data: Array(24).fill(0),
                borderColor: '#0284c7',
                backgroundColor: 'rgba(2, 132, 199, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top' }
            },
            scales: {
                x: { title: { display: true, text: 'Forecast Horizon (Next 24 Hours)' } },
                y: { title: { display: true, text: 'Predicted Liters / Hour' } }
            }
        }
    });
}

function appendLiveChartPoint(timestamp, actual, predicted) {
    if (!liveChartInstance) return;

    liveChartBuffer.labels.push(timestamp);
    liveChartBuffer.actual.push(actual);
    liveChartBuffer.predicted.push(predicted !== null ? predicted : actual);

    if (liveChartBuffer.labels.length > MAX_LIVE_BUFFER) {
        liveChartBuffer.labels.shift();
        liveChartBuffer.actual.shift();
        liveChartBuffer.predicted.shift();
    }

    liveChartInstance.data.labels = liveChartBuffer.labels;
    liveChartInstance.data.datasets[0].data = liveChartBuffer.actual;
    liveChartInstance.data.datasets[1].data = liveChartBuffer.predicted;
    liveChartInstance.update();
}

function updateForecastChart(forecastValues) {
    if (!forecastChartInstance) return;
    forecastChartInstance.data.datasets[0].data = forecastValues;
    forecastChartInstance.update();
}

function addEventLog(status, message, type = "normal") {
    const box = document.getElementById("event-log-box");
    if (!box) return;

    const row = document.createElement("div");
    const timeStr = new Date().toLocaleTimeString();
    row.className = `log-row log-${type}`;
    row.innerHTML = `<span class="log-time">${timeStr}</span><span class="log-msg"><strong>[${status}]</strong> ${message}</span>`;

    box.insertBefore(row, box.firstChild);

    while (box.children.length > 50) {
        box.removeChild(box.lastChild);
    }
}

function clearEventLog() {
    const box = document.getElementById("event-log-box");
    if (box) {
        box.innerHTML = `<div class="log-row log-info"><span class="log-time">[System]</span><span class="log-msg">Event log cleared. Stream active.</span></div>`;
    }
}

function updateModelDetails(metrics) {
    if (!metrics) return;
    if (metrics.dnn) {
        document.getElementById("det-dnn-mae").innerText = `${metrics.dnn.MAE || '--'} L/hr`;
        document.getElementById("det-dnn-rmse").innerText = `${metrics.dnn.RMSE || '--'} L/hr`;
        document.getElementById("det-dnn-r2").innerText = metrics.dnn.R2 ? `${(metrics.dnn.R2 * 100).toFixed(1)}%` : "--";
    }
    if (metrics.lstm && !metrics.lstm.error) {
        document.getElementById("det-lstm-mae").innerText = `${metrics.lstm.MAE || '--'} L/hr`;
        document.getElementById("det-lstm-rmse").innerText = `${metrics.lstm.RMSE || '--'} L/hr`;
    }
    if (metrics.autoencoder) {
        document.getElementById("det-ae-thresh").innerText = metrics.autoencoder.threshold || "0.0083";
    }
    if (metrics.q_learning) {
        document.getElementById("det-q-states").innerText = metrics.q_learning.states_learned || "81";
    }
}

function roundVal(val, decimals = 1) {
    if (typeof val !== 'number') return val;
    return val.toFixed(decimals);
}
