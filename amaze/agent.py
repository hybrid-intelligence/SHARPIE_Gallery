import gymnasium as gym
from tamer import Tamer
import numpy as np

from environment import environment



class Agent:
    def __init__(self, name, room_name):
        self.name = name
        self.room_name = room_name
        self.tamer = Tamer(environment, model_file_to_load = room_name+'_model')

    def sample(self, observation):
        return self.tamer.act(observation, epsilon=0.1)
    
    def predict(self, observation):
        return self.tamer.act(observation)
    
    def train(self, state, action, reward, done, next_state):
        print(state, self.tamer.predict(state))
        if reward == 0 and not done: 
            return
        if not done:
            td_target = reward
            self.tamer.train(state, action, td_target)
        self.tamer.save_model(self.room_name+'_model')
    
    
def create_agents(room_name):
    return [Agent('agent_0', room_name)]