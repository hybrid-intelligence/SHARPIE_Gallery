import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gym
from gym import Env
from gym import Wrapper
import numpy as np

class EnvironmentWrapper(Wrapper):
    """Wrapper for the Gym Super Mario Bros environment."""

    def __init__(self):
        """Initialize the environment."""
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))
        
        # fields needed to adapt to gymnasium
        self.env.metadata['render_fps'] = self.env.metadata['video.frames_per_second']
        self.env.metadata['render_modes'] = self.env.metadata['render.modes'] 

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
        self._action_meanings = {0: "NOOP",
                                 128: "RIGHT",
                                 129: "RIGHT A",
                                 130: "RIGHT B",
                                 131: "RIGHT A B",
                                 1: "A",
                                 64: "LEFT"
                                }

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation (numpy array)
            info: Additional information
        """
        return self.env.reset(), {}     # info added for sharpie

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
        actions = list(action_dict.values())[0]
        given_actions = ['0'] * 4

        action = 0b00000000
        for act in actions:
            action |= act

        if action not in self._action_meanings:     #validity check
            action = 0  #NOOP if invalid

        state, reward, done, info = self.env.step(action)
        return state, reward, done, done, info          # repeating done for terminated/truncated

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (numpy array)
        """
        render_array = self.env.render(mode='rgb_array')
        return render_array[..., ::-1]      #opencv waits BGR order 


environment = EnvironmentWrapper()
