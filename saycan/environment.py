"""
SayCan Environment Wrapper for SHARPIE.

This module wraps the PickPlaceEnv from the SayCan codebase to work with the
SHARPIE experiment framework. It integrates:
- ViLD for open-vocabulary object detection
- LLM (via Ollama) for task planning and action scoring
- CLIPort for language-conditioned pick-and-place manipulation

Action Types:
- "task:<description>" - Set task and auto-plan first action
- "plan" - Get next planned action from LLM
- "<text instruction>" - Direct CLIPort instruction
- "done" - End episode

Original SayCan Repository:
    https://github.com/google-research/google-research/tree/master/saycan

Reference:
    Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, Y., David, B.,
    Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Ho, D.,
    Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jang, E., Jang, R., Julian, R.,
    ... & Zeng, A. (2022). Do As I Can, Not As I Say: Grounding Language in
    Robotic Affordances. arXiv preprint arXiv:2204.01691.
"""
import os 
import sys

# Add the saycan directory to path for imports
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))
if SAYCAN_DIR not in sys.path:
    sys.path.insert(0, SAYCAN_DIR)
    
from base_environment import SayCanBaseEnvironment

class EnvironmentWrapper(SayCanBaseEnvironment):
    """Wrapper for the SayCan PickPlaceEnv with LLM planning and CLIPort integration."""

    def __init__(self):
        """Initialize the environment."""
        super().__init__()

    def step(self, action_dict):
        """
        Execute one step in the environment.

        Args:
            action_dict: Dictionary with agent id as keys and action as value.
                        Action can be:
                        - string text instruction directly
                        - "task:<description>" to set a task and auto-plan
                        - "plan" to get the next planned action
                        - "done" to end the episode

        Returns:
            observation: New observation dict
            reward: Reward for the action (float)
            terminated: Whether the episode has ended (bool)
            truncated: Whether the episode was truncated (bool)
            info: Additional information (dict)
        """
        # Extract action from dict (single-agent environment)
        action = list(action_dict.values())[0] if isinstance(action_dict, dict) else action_dict
        return super().step(action)

# Create the environment instance for SHARPIE runner
environment = EnvironmentWrapper()
