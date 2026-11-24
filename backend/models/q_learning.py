"""
Tabular Q-Learning Agent for Intelligent Water Pump Control.
Reinforcement Learning implementation for automated decision-making.
"""

import os
import json
import random
import numpy as np
from backend.utils.config import Q_TABLE_PATH

class QLearningPumpAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Parameters:
        -----------
        alpha : float
            Learning rate
        gamma : float
            Discount factor for future rewards
        epsilon : float
            Exploration probability for epsilon-greedy selection
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [0, 1, 2] # 0 = OFF, 1 = ON 5 min, 2 = ON 10 min
        self.action_labels = {
            0: "Pump OFF",
            1: "Pump ON for 5 minutes",
            2: "Pump ON for 10 minutes"
        }
        self.q_table = {}

    def discretize_state(self, tank_level, predicted_demand, current_consumption, leakage_risk):
        """
        Categorizes numerical inputs into discrete categorical states.
        
        State space dimensions:
        - tank_level: LOW (< 30%), MEDIUM (30-70%), HIGH (> 70%)
        - predicted_demand: LOW (< 300L), MEDIUM (300-800L), HIGH (> 800L)
        - current_consumption: LOW (< 300L), MEDIUM (300-800L), HIGH (> 800L)
        - leakage_risk: LOW (< 0.3), MEDIUM (0.3-0.7), HIGH (> 0.7)
        """
        # Tank level category
        if tank_level < 30.0:
            t_cat = "LOW"
        elif tank_level <= 70.0:
            t_cat = "MEDIUM"
        else:
            t_cat = "HIGH"
            
        # Predicted demand category
        if predicted_demand < 300.0:
            d_cat = "LOW"
        elif predicted_demand <= 800.0:
            d_cat = "MEDIUM"
        else:
            d_cat = "HIGH"
            
        # Current consumption category
        if current_consumption < 300.0:
            c_cat = "LOW"
        elif current_consumption <= 800.0:
            c_cat = "MEDIUM"
        else:
            c_cat = "HIGH"
            
        # Leakage risk category
        if leakage_risk < 0.3:
            r_cat = "LOW"
        elif leakage_risk <= 0.7:
            r_cat = "MEDIUM"
        else:
            r_cat = "HIGH"
            
        return (t_cat, d_cat, c_cat, r_cat)

    def get_q_values(self, state_tuple):
        """Returns Q-values list for all actions given a state key."""
        state_key = str(state_tuple)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]
        return self.q_table[state_key]

    def choose_action(self, state_tuple, training=False):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = self.get_q_values(state_tuple)
        return int(np.argmax(q_vals))

    def calculate_reward(self, state_tuple, action):
        """
        Reward Function R(s, a):
        
        Positive rewards:
        + 10 for maintaining ideal tank level (MEDIUM/HIGH) when demand is high.
        + 5 for turning pump OFF when tank is HIGH and demand is LOW.
        
        Negative rewards / Penalties:
        - 20 for tank LOW and pump OFF (Risk of dry running / water shortage).
        - 25 for tank HIGH and pump ON 10 min (Risk of overflow).
        - 30 for high leakage risk and turning pump ON (Wasting water during leak).
        """
        t_cat, d_cat, c_cat, r_cat = state_tuple
        reward = 0.0
        
        # High leakage risk penalty
        if r_cat == "HIGH" and action > 0:
            reward -= 30.0
        elif r_cat == "HIGH" and action == 0:
            reward += 15.0 # Wise decision to keep pump off during suspected pipe burst
            
        # Tank low penalty / reward
        if t_cat == "LOW":
            if action == 2:
                reward += 20.0 # Great response to low water
            elif action == 1:
                reward += 10.0
            else:
                reward -= 25.0 # Water shortage hazard!
                
        # Tank high penalties
        elif t_cat == "HIGH":
            if action == 0:
                reward += 10.0 # Prevents overflow
            elif action == 1:
                reward -= 10.0
            else: # Action 2
                reward -= 25.0 # Danger of overflow!
                
        # Tank medium scenarios
        elif t_cat == "MEDIUM":
            if d_cat == "HIGH" and action > 0:
                reward += 15.0 # Proactive filling for upcoming peak
            elif d_cat == "LOW" and action == 0:
                reward += 5.0
            else:
                reward += 2.0
                
        return reward

    def update_q(self, state_tuple, action, reward, next_state_tuple):
        """Applies Bellman Equation to update Q-table: Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]"""
        state_key = str(state_tuple)
        next_state_key = str(next_state_tuple)
        
        current_q = self.get_q_values(state_tuple)[action]
        max_next_q = max(self.get_q_values(next_state_tuple))
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = round(float(new_q), 3)

    def train_agent(self, episodes=5000):
        """Simulates environment episodes to learn optimal control policy."""
        t_levels = ["LOW", "MEDIUM", "HIGH"]
        demands = ["LOW", "MEDIUM", "HIGH"]
        consumptions = ["LOW", "MEDIUM", "HIGH"]
        risks = ["LOW", "MEDIUM", "HIGH"]
        
        for _ in range(episodes):
            state = (
                random.choice(t_levels),
                random.choice(demands),
                random.choice(consumptions),
                random.choice(risks)
            )
            action = self.choose_action(state, training=True)
            reward = self.calculate_reward(state, action)
            
            # Simple next state transition simulation
            next_t = state[0]
            if action == 2:
                next_t = "HIGH" if state[0] == "MEDIUM" else ("MEDIUM" if state[0] == "LOW" else "HIGH")
            elif action == 0:
                next_t = "LOW" if state[0] == "MEDIUM" else ("MEDIUM" if state[0] == "HIGH" else "LOW")
                
            next_state = (next_t, random.choice(demands), random.choice(consumptions), random.choice(risks))
            self.update_q(state, action, reward, next_state)
            
        self.save()
        print(f"[+] Q-Learning agent trained for {episodes} episodes. States learned: {len(self.q_table)}")

    def recommend(self, tank_level, predicted_demand, current_consumption, leakage_risk):
        """Generates pump control action recommendation and student-explainable rationale."""
        if not self.q_table:
            self.load()
            
        state_tuple = self.discretize_state(tank_level, predicted_demand, current_consumption, leakage_risk)
        action = self.choose_action(state_tuple, training=False)
        action_name = self.action_labels[action]
        reward = self.calculate_reward(state_tuple, action)
        
        # Explainable reasoning generator
        t_cat, d_cat, c_cat, r_cat = state_tuple
        reason_parts = []
        
        if r_cat == "HIGH":
            reason_parts.append("Elevated leakage risk detected (suspected pipe breach).")
        if t_cat == "LOW":
            reason_parts.append("Storage tank level is low.")
        elif t_cat == "HIGH":
            reason_parts.append("Storage tank is near maximum capacity.")
        if d_cat == "HIGH":
            reason_parts.append("Predicted demand is high.")
        elif d_cat == "LOW":
            reason_parts.append("Predicted demand is low.")
            
        if action == 0:
            if r_cat == "HIGH":
                reason = "Pump kept OFF to prevent water loss from active leak."
            elif t_cat == "HIGH":
                reason = "Pump kept OFF as tank level is already sufficient to avoid overflow."
            else:
                reason = "Pump kept OFF as current water reserve matches demand."
        elif action == 1:
            reason = "Pump turned ON for 5 minutes for moderate tank replenishment."
        else: # Action 2
            reason = "Pump turned ON for 10 minutes to prevent impending water shortage."
            
        return {
            "state": {
                "tank_level": t_cat,
                "predicted_demand": d_cat,
                "current_consumption": c_cat,
                "leakage_risk": r_cat
            },
            "recommended_action": action_name,
            "action_id": action,
            "expected_reward": reward,
            "reason": reason
        }

    def save(self):
        """Saves Q-table to JSON file."""
        os.makedirs(os.path.dirname(Q_TABLE_PATH), exist_ok=True)
        with open(Q_TABLE_PATH, 'w') as f:
            json.dump(self.q_table, f, indent=2)

    def load(self):
        """Loads Q-table from JSON file."""
        if os.path.exists(Q_TABLE_PATH):
            with open(Q_TABLE_PATH, 'r') as f:
                self.q_table = json.load(f)
            return True
        return False
