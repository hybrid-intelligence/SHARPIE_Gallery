import pickle
from pathlib import Path

import numpy as np

import environment

from overcooked_ai_py.agents.agent import RandomAgent

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class Agent:
    def __init__(self, name, room_name):
        self.name = name
        self.room_name = room_name
        self.env = environment.environment
        if (model_file_to_load := MODELS_DIR.joinpath(room_name + '_model.p')).exists():
            self.load_model(filename=model_file_to_load)
        else:
            self.policy = RandomAgent(self.env)

    def sample(self, observation):
        return self.env.action_space.sample()
    
    def predict(self, observation):
        # return self.H.predict([state])
        return self.policy.predict(observation)

    @staticmethod
    def argmax_random(x):
        _max = x.max()
        maxes = np.flatnonzero(x == _max)
        return np.random.choice(maxes)

    def act(self, state, epsilon=0.0):
        if np.random.random() < 1 - epsilon:
            print(self.predict(state))
            return self.argmax_random(self.predict(state))
        else:
            return np.random.randint(0, self.env.action_space.n)

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            self.H = pickle.load(f)

    
def create_agents(room_name):
    return [Agent('agent', room_name)]
