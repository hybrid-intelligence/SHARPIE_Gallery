import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np

class EnvironmentWrapper():
    """Wrapper for the Gym Super Mario Bros environment."""

    def __init__(self):
        """Initialize the environment."""
        env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='rgb_array')
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=self.env.observation_space.shape,
                dtype=np.uint8)

        self.action_space = gym.spaces.Discrete(len(SIMPLE_MOVEMENT))
        
        # Binary keyboard inputs → discrete index mapping
        self.binary_to_index = {
            0: 0,      # NOOP
            1: 5,      # A (jump)
            2: 0,      # B alone → NOOP (not in SIMPLE_MOVEMENT)
            64: 6,     # left
            128: 1,    # right
            129: 2,    # right + A
            130: 3,    # right + B
            131: 4,    # right + A + B
        }
        
        # Action names for display (index → name)
        self.action_meanings = {i: name for i, name in enumerate([
            "NOOP",
            "RIGHT",
            "RIGHT A",
            "RIGHT B",
            "RIGHT A B",
            "A",
            "LEFT"
        ])}

        self.metadata = self.env.metadata
        self.last_transition = ()

    def translate_action(self, given_actions):
        """
        Convert binary keyboard inputs to discrete action index.
        
        Args:
            given_actions: list of binary button values
        
        Returns:
            int: discrete action index (0-6)
        """
        binary_action = 0
        for act in given_actions:
            binary_action |= act
        
        if binary_action in self.binary_to_index:
            return self.binary_to_index[binary_action]
        
        # Handle unknown combinations
        if binary_action & 64:
            return 6  # left
        if binary_action & 128:
            return 1  # right
        return 0  # NOOP

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
        given_actions = list(action_dict.values())[0]
        action_index = self.translate_action(given_actions)
        
        next_obs, reward, terminated, truncated, info = self.env.step(action_index)

        self.last_transition = (next_obs, reward, terminated or truncated, info)

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