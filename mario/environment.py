import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gymnasium.wrappers import EnvCompatibility
import gymnasium

import gym as old_gym
import numpy as np

class EnvironmentWrapper():
    """Wrapper for the Gym Super Mario Bros environment."""

    def __init__(self):
        """Initialize the environment."""
        self.env = EnvCompatibility(gym_super_mario_bros.make('SuperMarioBros-v0'), "rgb_array")
        
        self.observation_space = gymnasium.spaces.Box(
                low = self.env.observation_space.low,
                high = self.env.observation_space.high,
                shape = self.env.observation_space.shape,
                dtype = self.env.observation_space.dtype)

        self.action_space = gymnasium.spaces.Discrete(len(SIMPLE_MOVEMENT))
        
        # fields needed to adapt to gymnasium
        self.env.metadata['render_fps'] = self.env.metadata['video.frames_per_second']
        self.env.metadata['render_modes'] = self.env.metadata['render.modes'] 

        self.last_transition = ()

        """
        complete button map of mario environment
        _button_map = {
            'right':  0b10000000,
            'left':   0b01000000,
            'down':   0b00100000,
            'up':     0b00010000,
            'start':  0b00001000,
            'select': 0b00000100,
            'B':      0b00000010,       #SPEED
            'A':      0b00000001,       #JUMP
            'NOOP':   0b00000000,
        }
        """

        # valid actions for SIMPLE_MOVEMENT
        self.action_meanings = {0: "NOOP",
                                128: "RIGHT",
                                129: "RIGHT A",
                                130: "RIGHT B",
                                131: "RIGHT A B",
                                1: "A",
                                64: "LEFT"
                               }

        self.metadata = self.env.metadata

    def translate_action(self, given_actions):
        action = 0b00000000
        for act in given_actions:
            action |= act

        if action not in self.action_meanings:     #validity check
            action = 0  #NOOP if invalid
        
        return action

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation (numpy array)
            info: Additional information
        """
        return self.env.reset()

    def step(self, action_dict):
        """
        Execute one step in the environment.

        Args:
            action_dict: Dictionary with agent id as keys and action as value

        Returns:
            observation: New observation (numpy array)
            reward: Reward for the action (float)
            terminated: Whether the episode has ended (bool)
            truncated: Whether the episode was truncated (bool)
            info: Additional information (dict)
        """
        # Convert dict action to discrete actions ([int]) - Mario is single-agent / multi inputs
        given_actions = list(action_dict.values())[0]
        action = self.translate_action(given_actions)
        
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.last_transition = (next_obs, reward, terminated or truncated, info)    # save the last transition

        return next_obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (numpy array)
        """
        render_array = self.env.render()
        return render_array[..., ::-1]      #opencv waits BGR order 

    def close(self):
        pass

environment = EnvironmentWrapper()
