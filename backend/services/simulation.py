"""
Real-Time Data Simulation Engine for AI WaterWise.
Generates real-time telemetry ticks under 5 distinct operational simulation conditions,
evaluating each tick dynamically against DNN, Autoencoder, Bayesian Risk, and Q-Agent models.
"""

import time
import math
import random
from datetime import datetime

from backend.services.preprocessing import DataPreprocessor
from backend.models.dnn_model import DNNConsumptionModel
from backend.models.lstm_model import LSTMForecastingModel
from backend.models.autoencoder import AutoencoderAnomalyDetector
from backend.services.bayesian import BayesianLeakageEstimator
from backend.services.recommendation import PumpRecommendationService
from backend.utils.config import verify_models_for_current_dataset, get_active_csv_path

class SimulationEngine:
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.condition = "normal"  # normal, major_leak, micro_leak, sensor_fault, heatwave_surge
        self.speed = 1.0  # seconds per tick
        self.tick_count = 0
        self.history = []
        self.max_history = 100
        
        self.preprocessor = DataPreprocessor()
        self.dnn_model = DNNConsumptionModel()
        self.lstm_model = LSTMForecastingModel()
        self.ae_model = AutoencoderAnomalyDetector()
        self.bayes_estimator = BayesianLeakageEstimator()
        self.rec_service = PumpRecommendationService()
        
        self.df_data = None
        self.csv_index = 0
        self.simulated_tank_level = 75.0
        
    def _ensure_data_loaded(self):
        """Loads baseline dataset or initializes default synthetic dataframe."""
        if self.df_data is None or len(self.df_data) == 0:
            try:
                self.df_data = self.preprocessor.load_data()
            except Exception:
                self.df_data = None

    def start(self, condition: str = "normal", speed: float = 1.0):
        """Starts real-time simulation with given condition and speed."""
        valid_conds = ["normal", "major_leak", "micro_leak", "sensor_fault", "high_demand", "low_tank"]
        self.condition = condition if condition in valid_conds else "normal"
        self.speed = max(0.1, min(5.0, speed))
        self.is_running = True
        self.is_paused = False
        self._ensure_data_loaded()
        return self.get_status()

    def pause(self):
        """Pauses simulation playback."""
        if self.is_running:
            self.is_paused = True
        return self.get_status()

    def resume(self):
        """Resumes paused simulation playback."""
        if self.is_running:
            self.is_paused = False
        return self.get_status()

    def stop(self):
        """Stops simulation playback and resets state."""
        self.is_running = False
        self.is_paused = False
        self.tick_count = 0
        self.history.clear()
        self.simulated_tank_level = 75.0
        return self.get_status()

    def set_condition(self, condition: str):
        """Dynamically switches simulation condition on-the-fly."""
        valid_conds = ["normal", "major_leak", "micro_leak", "sensor_fault", "high_demand", "low_tank"]
        if condition in valid_conds:
            self.condition = condition
        return self.get_status()

    def get_status(self):
        """Returns current engine status summary."""
        state_str = "STOPPED"
        if self.is_running:
            state_str = "PAUSED" if self.is_paused else "RUNNING"
            
        return {
            "state": state_str,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "condition": self.condition,
            "speed": self.speed,
            "tick_count": self.tick_count,
            "simulated_tank_level": round(self.simulated_tank_level, 2),
            "history_length": len(self.history)
        }

    def generate_next_tick(self):
        """
        Synthesizes the next real-time telemetry tick according to active condition
        and passes it through the AI pipeline.
        """
        self._ensure_data_loaded()
        self.tick_count += 1
        hour_of_day = (self.tick_count % 24)
        current_time_str = datetime.now().strftime("%H:%M:%S")
        
        # 1. Base telemetry parameters from CSV seed or dynamic physical model
        if self.df_data is not None and len(self.df_data) > 0:
            idx = self.csv_index % len(self.df_data)
            row = self.df_data.iloc[idx].to_dict()
            self.csv_index += 1
            
            temp = float(row.get('temperature', 25.0))
            humidity = float(row.get('humidity', 65.0))
            occupancy = int(row.get('occupancy', 20))
            day_of_week = int(row.get('day_of_week', 1))
            is_holiday = int(row.get('is_holiday', 0))
            base_consumption = float(row.get('water_consumption', 250.0))
        else:
            # Fallback baseline synthesis if no CSV is uploaded
            is_night = (hour_of_day < 6 or hour_of_day > 22)
            occupancy = random.randint(0, 3) if is_night else random.randint(15, 40)
            temp = round(20.0 + 8.0 * math.sin(math.pi * hour_of_day / 12.0) + random.uniform(-1, 1), 2)
            humidity = round(75.0 - 15.0 * math.sin(math.pi * hour_of_day / 12.0) + random.uniform(-2, 2), 2)
            day_of_week = 2
            is_holiday = 0
            base_consumption = round((occupancy * 8.5) + (temp * 3.2) + random.uniform(10, 30), 2)

        prev_consumption = self.history[-1]['water_consumption'] if self.history else base_consumption

        # 2. Inject Operational Simulation Condition Effects
        sensor_fault_flag = False

        if self.condition == "normal":
            water_consumption = base_consumption
            self.simulated_tank_level = max(30.0, min(95.0, self.simulated_tank_level + random.uniform(-1.5, 1.5)))

        elif self.condition == "major_leak":
            # Massive abrupt consumption spike + fast tank decay
            water_consumption = base_consumption + random.uniform(800.0, 1200.0)
            self.simulated_tank_level = max(5.0, self.simulated_tank_level - random.uniform(4.0, 8.0))

        elif self.condition == "micro_leak":
            # Low / zero occupancy (night flow leakage pattern)
            occupancy = max(0, min(2, occupancy))
            water_consumption = 65.0 + random.uniform(15.0, 35.0)
            self.simulated_tank_level = max(10.0, self.simulated_tank_level - random.uniform(0.8, 1.5))

        elif self.condition == "sensor_fault":
            water_consumption = base_consumption
            temp = 68.5  # Out of bounds sensor reading
            humidity = 2.0
            sensor_fault_flag = True

        elif self.condition == "high_demand":
            occupancy = max(75, occupancy + 35)
            water_consumption = (occupancy * 11.5) + (temp * 3.5) + random.uniform(30, 70)
            self.simulated_tank_level = max(15.0, self.simulated_tank_level - random.uniform(2.5, 4.5))

        elif self.condition == "low_tank":
            water_consumption = base_consumption
            self.simulated_tank_level = max(5.0, self.simulated_tank_level - random.uniform(3.0, 6.0))

        water_consumption = round(water_consumption, 2)
        tank_level = round(self.simulated_tank_level, 2)

        # 3. AI Pipeline Evaluation
        dnn_features = {
            'temperature': temp,
            'humidity': humidity,
            'occupancy': occupancy,
            'day_of_week': day_of_week,
            'is_holiday': is_holiday,
            'tank_level': tank_level,
            'prev_consumption': prev_consumption
        }

        ae_features = {
            'temperature': temp,
            'humidity': humidity,
            'occupancy': occupancy,
            'tank_level': tank_level,
            'water_consumption': water_consumption
        }

        models_trained = verify_models_for_current_dataset()

        # A. DNN Demand Prediction
        if models_trained and self.dnn_model.load():
            predicted_consumption = self.dnn_model.predict(dnn_features)
        else:
            predicted_consumption = round((occupancy * 8.5) + (temp * 3.2) + 20.0, 2)

        pred_diff = round(water_consumption - predicted_consumption, 2) if predicted_consumption is not None else 0.0

        # B. Autoencoder Anomaly Detection
        if models_trained and self.ae_model.load():
            ae_res = self.ae_model.detect(ae_features)
        else:
            dev = abs(water_consumption - (predicted_consumption or 250.0))
            is_anom = dev > 300.0 or sensor_fault_flag or (self.condition == "micro_leak" and occupancy <= 2 and water_consumption > 50)
            rec_err = round(dev / 10.0, 4)
            ae_res = {
                "reconstruction_error": rec_err,
                "threshold": 0.0083,
                "status": "ANOMALY DETECTED" if is_anom else "NORMAL",
                "is_anomaly": is_anom
            }

        anomaly_status_str = "ANOMALY DETECTED" if ae_res.get('is_anomaly', False) else "NORMAL"

        # C. Bayesian Leakage Risk Estimation
        bayes_res = self.bayes_estimator.calculate_leakage_probability(
            is_anomaly=ae_res.get('is_anomaly', False),
            consumption=water_consumption,
            occupancy=occupancy,
            tank_level=tank_level,
            temperature=temp
        )

        leak_prob = bayes_res['probability']
        leak_pct = bayes_res['percentage']

        # Construct Bayesian evidence points
        leak_evidence = [
            f"Actual consumption ({water_consumption} L/hr) vs expected ({predicted_consumption} L/hr)",
            f"Occupancy: {occupancy} people",
            f"Tank level: {tank_level}%",
            f"Anomaly status: {anomaly_status_str}",
            f"Recent flow trend: {'Abrupt spike' if self.condition == 'major_leak' else ('Persistent night flow' if self.condition == 'micro_leak' else 'Steady')}"
        ]

        # D. Q-Agent Pump Recommendation
        if models_trained:
            pred_val = predicted_consumption if predicted_consumption is not None else water_consumption
            rec_res = self.rec_service.get_pump_recommendation(
                tank_level=tank_level,
                predicted_demand=float(pred_val),
                current_consumption=water_consumption,
                leakage_probability=leak_prob
            )
        else:
            rec_action = "INVESTIGATE LEAK" if leak_prob > 0.7 else ("PUMP ON — 10 MIN" if tank_level < 30 else "PUMP OFF")
            rec_res = {
                "recommended_action": rec_action,
                "reason": f"Real-time rule response based on tank level ({tank_level}%) and leakage risk ({leak_pct}%)."
            }

        # Build Q-State Representation
        tank_state = "LOW" if tank_level < 30 else ("MEDIUM" if tank_level < 70 else "HIGH")
        demand_state = "HIGH" if (predicted_consumption or 0) > 300 else "NORMAL"
        risk_state = "HIGH" if leak_prob > 0.6 else ("MEDIUM" if leak_prob > 0.3 else "LOW")
        anomaly_state = "YES" if anomaly_status_str == "ANOMALY DETECTED" else "NO"

        q_state_dict = {
            "tank": tank_state,
            "demand": demand_state,
            "risk": risk_state,
            "anomaly": anomaly_state
        }

        # Generate 24h LSTM forecast list
        lstm_forecast = []
        if models_trained and self.lstm_model.load():
            try:
                hist_vals = [t['water_consumption'] for t in self.history[-24:]]
                while len(hist_vals) < 24:
                    hist_vals.insert(0, water_consumption)
                lstm_forecast = self.lstm_model.forecast_next_hours(hist_vals, horizon=24)
            except Exception:
                pass

        if not lstm_forecast:
            # Fallback 24h diurnal forecast curve based on predicted demand
            base_p = predicted_consumption or 200.0
            lstm_forecast = [
                round(base_p * (1.0 + 0.3 * math.sin(2 * math.pi * (hour_of_day + h) / 24.0)), 1)
                for h in range(1, 25)
            ]

        cond_label = self._get_condition_label(self.condition)

        tick_payload = {
            "tick": self.tick_count,
            "timestamp": current_time_str,
            "temperature": temp,
            "humidity": humidity,
            "occupancy": occupancy,
            "tank_level": tank_level,
            "water_consumption": water_consumption,
            "actual_consumption": water_consumption,
            "predicted_consumption": predicted_consumption,
            "prediction_difference": pred_diff,
            "lstm_forecast": lstm_forecast,
            "anomaly_status": anomaly_status_str,
            "anomaly_type": cond_label,
            "reconstruction_error": ae_res['reconstruction_error'],
            "anomaly_threshold": ae_res.get('threshold', 0.0083),
            "is_anomaly": ae_res.get('is_anomaly', False),
            "leakage_probability": leak_prob,
            "leakage_percentage": leak_pct,
            "leakage_explanation": bayes_res['explanation'],
            "leakage_evidence": leak_evidence,
            "q_state": q_state_dict,
            "q_action": rec_res['recommended_action'],
            "recommended_action": rec_res['recommended_action'],
            "q_value": round(float(rec_res.get('confidence', 0.95)), 2),
            "q_reason": rec_res['reason'],
            "recommendation_reason": rec_res['reason'],
            "scenario": self.condition,
            "condition": self.condition,
            "condition_label": cond_label
        }

        # Store in history buffer
        self.history.append(tick_payload)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return tick_payload

    @staticmethod
    def _get_condition_label(condition):
        labels = {
            "normal": "NORMAL",
            "major_leak": "PIPE BURST",
            "micro_leak": "MICRO LEAK",
            "sensor_fault": "SENSOR FAULT",
            "high_demand": "HIGH DEMAND",
            "low_tank": "LOW TANK"
        }
        return labels.get(condition, condition.upper())

# Global singleton engine instance
simulation_engine = SimulationEngine()
