from pathlib import Path
from environment import EnvironmentWrapper
from imitation.algorithms import bc
from imitation.data.types import TransitionsWithRew
import numpy as np

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')

from stable_baselines3.ppo import MlpPolicy


class HumanExpert:
    
    def __init__(self, model_file_to_load = None):
        self.env = EnvironmentWrapper()
        self.rng = np.random.default_rng(0)

        self.bc_trainer = None

        
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.dones = []
        self.infos = []


    def train(self):
        # batch training from saved transitions
        transitions = TransitionsWithRew(obs = np.stack(self.observations),
                                  acts = np.array([int(a) if np.isscalar(a) else np.array(a) for a in self.actions]),
                                  next_obs = np.stack(self.next_observations),
                                  dones = np.array(self.dones),
                                  infos = np.array(self.infos),
                                  rews = np.array(self.rewards))
        
        policy = MlpPolicy(
           observation_space=self.env.observation_space,
           action_space=self.env.action_space,
           lr_schedule=lambda _: 1e-3
        )

        self.bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=transitions,
            policy=policy,
            rng = self.rng,
        )

        self.bc_trainer.train(n_epochs=1)

    def add_transition(self, observation, action, next_observation, reward, done, info):
        self.observations.append(observation)
        self.actions.append(action)
        self.next_observations.append(next_observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def save_policy(self, filename):
        filename = filename + '.p' if not filename.endswith('.p') else filename
        filepath = MODELS_DIR.joinpath(filename)
        self.bc_trainer.policy.save(filepath)
