"""
Recommendation Service for AI WaterWise
Orchestrates predictions from DNN, LSTM, Autoencoder, and Bayesian Risk model into Q-learning pump recommendation.
"""

from backend.models.q_learning import QLearningPumpAgent

class PumpRecommendationService:
    def __init__(self):
        self.q_agent = QLearningPumpAgent()

    def get_pump_recommendation(self, tank_level, predicted_demand, current_consumption, leakage_probability):
        """
        Calculates optimal pump control action using the Q-learning policy.
        """
        result = self.q_agent.recommend(
            tank_level=tank_level,
            predicted_demand=predicted_demand,
            current_consumption=current_consumption,
            leakage_risk=leakage_probability
        )
        return result
