from pathlib import Path

from overcooked_ai_py.agents.agent import RandomAgent

import environment

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')


class Policy:
    def __init__(self, id=""):
        self.id = id
        self.env = environment.environment

    def sample(self):
        return self.env.action_space.sample()
    
    def predict(self, observation, participant_input=None):
        return self.sample()


policy = Policy()
