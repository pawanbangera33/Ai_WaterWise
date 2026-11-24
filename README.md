# AI WaterWise: Intelligent Water Demand Forecasting and Resource Management System

A 7th-Semester Engineering College Project for **Advanced AI & ML** and **Neural Networks & Deep Learning**.

---

## 📋 System Pipeline Architecture

```
User CSV
    ↓
Validation
    ↓
Preprocessing
    ↓
DNN
    ↓
LSTM
    ↓
Autoencoder
    ↓
Bayesian reasoning
    ↓
Q-learning
    ↓
Dashboard
```

### Detailed Dataflow Workflow

1. **User CSV**: User uploads water-consumption CSV dataset through the web portal.
2. **Validation**: Backend (`POST /api/upload`) checks file type (.csv), required columns, empty rows, and structure.
3. **Preprocessing**: Backend (`POST /api/process`) sorts chronologically by timestamp, handles missing values, and builds model feature representations.
4. **DNN (Deep Neural Network)**: Feed-forward neural network predicts expected water consumption based on environmental parameters.
5. **LSTM (Long Short-Term Memory)**: Recurrent neural network forecasts future 24-hour demand time-series.
6. **Autoencoder**: Encoder-decoder neural network reconstructs features to detect anomalies.
7. **Bayesian Reasoning**: Combines autoencoder anomaly flag and physical evidence to compute posterior leakage probability $P(\text{Leakage} \mid \text{Evidence})$.
8. **Q-Learning Agent**: Evaluates discretized state space (tank level, demand, flow, leakage risk) to select optimal pump action (0 = Pump OFF, 1 = Pump ON for 5 minutes, 2 = Pump ON for 10 minutes).
9. **Dashboard**: Renders current readings (from latest valid CSV row), AI predictions, anomaly status, Q-agent recommendations, and 3 Chart.js graphs.

---

## 📊 Required CSV Dataset Format

The uploaded CSV dataset must contain the following **8 required columns**:

| Column Name | Data Type | Description | Example Value |
|---|---|---|---|
| `timestamp` | Datetime / String | Hourly timestamp (`YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DD HH:MM`) | `2026-09-20 08:00` |
| `temperature` | Float / Numeric | Outdoor ambient temperature (°C) | `27` |
| `humidity` | Float / Numeric | Relative humidity percentage (%) | `78` |
| `occupancy` | Integer | Active building occupant count | `82` |
| `day_of_week` | String / Integer | Day of week (`Sunday` or `0`–`6`) | `Sunday` |
| `is_holiday` | Integer | Public holiday flag (`1` = Yes, `0` = No) | `0` |
| `tank_level` | Float / Numeric | Overhead storage tank level percentage (`0`–`100`%) | `72` |
| `water_consumption` | Float / Numeric | Observed water consumption in Liters / Hour | `110` |

### Example CSV Format

```csv
timestamp,temperature,humidity,occupancy,day_of_week,is_holiday,tank_level,water_consumption
2026-09-20 08:00,27,78,82,Sunday,0,72,110
2026-09-20 09:00,28,76,91,Sunday,0,66,121
2026-09-20 10:00,29,74,105,Sunday,0,61,134
```

---

## 🛠️ Complete Operational Workflows

### 1. Upload Workflow (`POST /api/upload`)
- Select a `.csv` file in the web portal and click **"1. Upload CSV"**.
- Backend validates file extension, required columns, and non-empty content.
- Saves file to `data/uploads/<filename>` (without overwriting existing files).
- Updates active tracking file `data/active_dataset.json`.

### 2. Processing Workflow (`POST /api/process`)
- Click **"2. Process Data"**.
- Backend parses timestamps, handles missing values, sorts chronologically, and creates feature matrices for DNN, LSTM time-series sequences, Autoencoder features, Bayesian factors, and Q-learning states.

### 3. Model Training Workflow (`POST /api/train`)
- Click **"3. Train Models"**.
- Backend trains the 4 model components (DNN, LSTM, Autoencoder, Q-learning agent) on the uploaded CSV data.
- Saves model weight artifacts, scalers, and metric summaries under `saved_models/`.

### 4. Prediction Workflow (`GET /api/dashboard`)
- Extracts the latest valid timestamped row of the uploaded CSV as the current system reading.
- Passes current state to DNN, LSTM, Autoencoder, Bayesian estimator, and Q-learning agent.
- Returns predictions and recommendations to update KPI cards, alerts, and Chart.js graphs.

---

## ⚙️ Backend API Reference (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Accepts and validates user CSV upload, saving to `data/uploads/` |
| `/api/process` | POST | Preprocesses, cleans, and sorts active uploaded dataset |
| `/api/train` | POST | Trains DNN, LSTM, Autoencoder, and Q-Agent on uploaded dataset |
| `/api/dataset-status` | GET | Returns metadata and state of active uploaded dataset |
| `/api/dashboard` | GET | Returns latest reading, model predictions, and recommendations |
| `/api/history` | GET | Returns historical water data for Chart.js line graphs |
| `/api/forecast` | GET | Returns multi-step demand forecast array from LSTM |
| `/api/anomaly` | GET | Returns Autoencoder loss score and Bayesian leakage risk |
| `/api/recommendation` | GET | Returns Q-learning pump recommendation and reasoning |
| `/api/predict` | POST | Executes live inference for custom parameter inputs |

---

## 🚀 How to Run the Application

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Start FastAPI Backend
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Access Web Interface
Open browser to **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** to interact with the system.
