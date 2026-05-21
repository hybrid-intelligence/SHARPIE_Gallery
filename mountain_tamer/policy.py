import gymnasium as gym
from tamer import TAMERAgent
import numpy as np
import os
from pathlib import Path


MODELS_DIR = Path(__file__).parent.joinpath('saved_models')


class Policy:
    def __init__(self, id=""):
        self.id = id
        self.tamer = TAMERAgent(state_dim=2, action_dim=3, learning_rate=0.1, discount_factor=0.9)
        path = MODELS_DIR.joinpath(f"{self.id}_tamer_model.npy") if self.id else None
        if path and os.path.exists(path):
            self.tamer.load_model(path)
        else:
            print(f"No existing model found for {self.id}. Starting with a new model." if self.id else "Starting with a new model.")

    def predict(self, observation, participant_input=None):
        action = self.tamer.select_action(observation)
        return action
    
    def update(self, state, action, reward, done, next_state):
        td_target = reward
        if reward is not None and reward != 0:
            self.tamer.update_reward_model(state, action, td_target)
            if self.id:
                path = MODELS_DIR.joinpath(f"{self.id}_tamer_model.npy")
                self.tamer.save_model(path)
    
# Create an instance of the policy for use by the runner
policy = Policy(id='save')