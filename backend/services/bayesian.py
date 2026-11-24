"""
Bayesian Risk Estimation Service
Calculates P(Leakage | Observed Conditions) using explainable Bayes' Rule.
Designed for 7th-semester viva explanation.
"""

class BayesianLeakageEstimator:
    def __init__(self):
        # Prior probability of leakage in water distribution networks P(Leakage)
        self.prior_leakage = 0.05 # 5% baseline prior
        self.prior_no_leakage = 0.95

    def calculate_leakage_probability(self, is_anomaly, consumption, occupancy, tank_level, temperature):
        """
        Calculates posterior leakage probability P(L | E) using Bayes' Theorem:
        
            P(L | E) = [ P(E | L) * P(L) ] / P(E)
            
        where Evidence E combines:
        1. Autoencoder Anomaly Flag
        2. Night flow ratio (High consumption during low occupancy)
        3. Extreme consumption threshold (> 400 L/hr)
        
        Parameters:
        -----------
        is_anomaly : bool
            Whether Autoencoder flagged the current state as an anomaly
        consumption : float
            Current hourly consumption (L/hr)
        occupancy : int
            Current building occupancy count
        tank_level : float
            Current overhead tank level (%)
        temperature : float
            Ambient temperature (°C)
            
        Returns:
        --------
        dict:
            "probability": float (0.0 to 1.0)
            "percentage": str (e.g. "72%")
            "explanation": str
        """
        # Step 1: Calculate likelihood P(Evidence | Leakage) vs P(Evidence | No Leakage)
        likelihood_leakage = 1.0
        likelihood_no_leakage = 1.0
        
        evidence_factors = []
        
        # Factor 1: Autoencoder Anomaly
        if is_anomaly:
            likelihood_leakage *= 0.85   # 85% of leaks trigger ML anomalies
            likelihood_no_leakage *= 0.15 # Only 15% of normal operations trigger false positive ML anomalies
            evidence_factors.append("Autoencoder ML anomaly detected")
        else:
            likelihood_leakage *= 0.15
            likelihood_no_leakage *= 0.85
            
        # Factor 2: Low Occupancy + High Consumption (Unexplained Night Flow)
        if occupancy <= 10 and consumption > 250.0:
            likelihood_leakage *= 0.90   # Severe indicator of continuous leak/rupture
            likelihood_no_leakage *= 0.05
            evidence_factors.append("High flow rate during low occupancy (Night flow anomaly)")
        elif occupancy > 50 and consumption > 500.0:
            likelihood_leakage *= 0.60
            likelihood_no_leakage *= 0.40
            
        # Factor 3: Excessive Consumption (> 500 L/hr)
        if consumption > 500.0:
            likelihood_leakage *= 0.80
            likelihood_no_leakage *= 0.10
            evidence_factors.append("Water flow rate exceeds normal operational threshold (>500 L/hr)")
            
        # Step 2: Apply Bayes' Rule
        # Numerator = P(E | L) * P(L)
        numerator = likelihood_leakage * self.prior_leakage
        
        # Denominator P(E) = P(E | L) * P(L) + P(E | ~L) * P(~L)
        denominator = (likelihood_leakage * self.prior_leakage) + (likelihood_no_leakage * self.prior_no_leakage)
        
        # Posterior P(L | E)
        posterior = numerator / denominator if denominator > 0 else 0.05
        posterior = max(0.01, min(0.99, posterior))
        
        pct = round(posterior * 100, 1)
        
        if not evidence_factors:
            explanation = "Normal operating conditions. Low baseline probability of leak."
        else:
            explanation = f"Elevated leakage probability driven by: {', '.join(evidence_factors)}."
            
        return {
            "probability": round(posterior, 4),
            "percentage": f"{pct}%",
            "risk_level": "HIGH" if pct > 70 else ("MEDIUM" if pct > 35 else "LOW"),
            "explanation": explanation
        }
