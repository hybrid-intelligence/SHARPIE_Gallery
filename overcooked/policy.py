from pathlib import Path

from overcooked_ai_py.agents.agent import RandomAgent

import environment

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')


class Agent:
    def __init__(self):
        self.env = environment.environment
        room_name = "foo"
        if (model_file_to_load := MODELS_DIR.joinpath(room_name + '_model.p')).exists():
            self.load_model(filename=model_file_to_load)
        else:
            self.policy = RandomAgent(self.env)

    def sample(self):
        return self.env.action_space.sample()
    
    def predict(self, observation):
        # return self.H.predict([state])
        return self.sample()

policy = Agent()
