import gymnasium as gym
from tamer import Tamer
import numpy as np



env = gym.make('FrozenLake-v1', is_slippery=False)

# hyperparameters
discount_factor = 0.95
epsilon = 0.8  # vanilla Q learning actually works well with no random exploration
min_eps = 0.2
num_episodes = 2
tame = True  # set to false for vanilla Q learning

# set a timestep for training TAMER
# the more time per step, the easier for the human
# but the longer it takes to train (in real time)
# 0.2 seconds is fast but doable
tamer_training_timestep = 0.3



class Agent:
    def __init__(self, name):
        self.name = name
        self.tamer = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame, tamer_training_timestep, model_file_to_load='tamer_model')

    def sample(self, observation):
        return np.argmax(self.tamer.H.predict([observation]))
    
    def predict(self, observation):
        return np.argmax(self.tamer.H.predict([observation]))
    
    def train(self, state, action, reward, done, next_state):
        if reward == 0 and not done: 
            return
        if done:
            td_target = -1
        else:
            td_target = reward + self.tamer.discount_factor * np.max(self.tamer.H.predict([next_state]))
        self.tamer.H.update([state], action, td_target)
        self.tamer.save_model('tamer_model')
    
    

agents = [Agent('agent_0')]