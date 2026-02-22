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
import tempfile
import numpy as np
from PIL import Image

# Add the saycan directory to path for imports
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))
if SAYCAN_DIR not in sys.path:
    sys.path.insert(0, SAYCAN_DIR)

from pick_place_env import PickPlaceEnv
from config import PICK_TARGETS, PLACE_TARGETS
from cliport import get_cliport
# Import LLM and helpers for planning
from llm import make_options, gpt3_scoring, gpt3_context, termination_string
from helpers import normalize_scores, step_to_nlp, affordance_scoring
from vild import vild, category_name_string, vild_params


class EnvironmentWrapper:
    """Wrapper for the SayCan PickPlaceEnv with LLM planning and CLIPort integration."""

    def __init__(self):
        """Initialize the environment."""
        self.env = PickPlaceEnv()
        self.config = None
        self._step_count = 0
        self._max_steps = 100
        self._cliport = None
        self.cached_video_frames = []

        # LLM planning state
        self._current_task = None
        self._max_tasks = 10
        self._gpt3_prompt = None
        self._options = None
        self._found_objects = None
        self._task_step_count = 0

    def reset(self, config=None):
        """
        Reset the environment to an initial state.

        Args:
            config: Optional configuration dict with 'pick' and 'place' lists.
                   If None, uses default objects.

        Returns:
            observation: Initial observation dict with 'image', 'xyzmap', 'pick', 'place'
            info: Additional information dict
        """
        self._step_count = 0
        self.cached_video_frames = []

        # Reset LLM planning state
        self._current_task = None
        self._gpt3_prompt = None
        self._options = None
        self._found_objects = None
        self._task_step_count = 0

        if config is None:
            config = {'pick':  ['yellow block', 'blue block', 'red block'],
                      'place': ['blue bowl', 'red bowl']}

        self.config = config
        observation = self.env.reset(config)

        info = {
            "step": 0,
            "config": config,
            "pick_objects": config.get("pick", []),
            "place_objects": config.get("place", [])
        }

        return observation, info

    def set_task(self, task_text):
        """
        Set the current task from natural language.

        Args:
            task_text: Task instruction (e.g., "put all the blocks in different corners")
        """
        self._current_task = task_text
        self._gpt3_prompt = gpt3_context + "\n# " + task_text + "\n"
        self._task_step_count = 0
        self._found_objects = None
        self._options = None
        print(f"Environment: Task set to '{task_text}'")

    def detect_objects(self, observation=None):
        """
        Detect objects in the scene using ViLD.

        Args:
            observation: Observation dict with 'image'. If None, uses current observation.

        Returns:
            found_objects: List of detected object names
        """
        if observation is None:
            observation = self.env.get_observation()

        # Save image to temp file for ViLD
        image = observation['image']
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
            Image.fromarray(image).save(temp_path)

        try:
            # Run ViLD detection
            prompt_swaps = [('block', 'cube')]
            found_objects = vild(temp_path, category_name_string, vild_params,
                                plot_on=False, prompt_swaps=prompt_swaps)
            print(f"Environment: Detected objects: {found_objects}")
        finally:
            # Clean up temp file
            os.unlink(temp_path)

        return found_objects

    def plan_next_action(self, observation=None):
        """
        Plan the next action using LLM + affordance scoring.

        Args:
            observation: Current observation. If None, uses current observation.

        Returns:
            action_text: Natural language action instruction
            done: Whether the task is complete
        """
        if observation is None:
            observation = self.env.get_observation()

        # Detect objects if not already done
        if self._found_objects is None:
            self._found_objects = self.detect_objects(observation)

        # Create options if not already done
        if self._options is None:
            self._options = make_options(PICK_TARGETS, PLACE_TARGETS,
                                         termination_string=termination_string)

        # Calculate affordance scores based on detected objects
        affordance_scores = affordance_scoring(self._options, self._found_objects,
                                               block_name="box", bowl_name="circle",
                                               verbose=False)

        # Get LLM scores
        llm_scores, _ = gpt3_scoring(self._gpt3_prompt, self._options, verbose=True)

        # Combine scores
        combined_scores = {
            option: np.exp(llm_scores[option]) * affordance_scores[option]
            for option in self._options
        }
        combined_scores = normalize_scores(combined_scores)

        # Select best action
        selected_task = max(combined_scores, key=combined_scores.get)

        # Check for termination
        if selected_task == termination_string:
            print("Environment: Task completed (termination signal)")
            return "done", True

        # Update prompt for next step
        self._gpt3_prompt += selected_task + "\n"
        self._task_step_count += 1

        # Check max tasks limit
        if self._task_step_count >= self._max_tasks:
            print("Environment: Max steps reached")
            return "done", True

        # Convert to natural language
        action_text = step_to_nlp(selected_task)
        print(f"Environment: Step {self._task_step_count} - {action_text}")
        return action_text, False

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
        self._step_count += 1

        if len(self.cached_video_frames) > 0:
            return np.array([]), 0.0, False, False, {"info": "No action taken"}

        # Extract action from dict (single-agent environment)
        action = list(action_dict.values())[0] if isinstance(action_dict, dict) else action_dict

        # Handle different action types
        if action == 'done':
            return np.array([]), 0.0, True, False, {"info": "Task completed"}
        elif isinstance(action, str) and action.startswith('task:'):
            # Set task and plan first action
            task_text = action[5:].strip()
            self.set_task(task_text)
            action_text, task_done = self.plan_next_action()
            if task_done or action_text == "done":
                return np.array([]), 0.0, False, False, {"info": "Task completed"}
            action = action_text
        elif action == 'plan':
            # Get next planned action
            if self._current_task is None:
                return np.array([]), 0.0, False, False, {"info": "No task set"}
            # Re-detect objects since they may have moved
            self._found_objects = None
            action_text, task_done = self.plan_next_action()
            if task_done or action_text == "done":
                return np.array([]), 0.0, False, False, {"info": "Task completed"}
            action = action_text

        if action and action != "done":
            # Direct text instruction
            obs, reward, _, info = self._step_with_text(action)
            # Get the frames buffer
            self.cached_video_frames = self.env.cache_video
        else:
            return np.array([]), 0.0, False, False, {"info": "No action taken"}

        # Check termination conditions
        terminated = False
        truncated = self._step_count >= self._max_steps

        info["step"] = self._step_count
        info["max_steps"] = self._max_steps

        return obs, reward, terminated, truncated, info

    def _step_with_text(self, text):
        """Execute a step using CLIPort with text instruction."""
        if self._cliport is None:
            self._cliport = get_cliport()

        # Get current observation
        obs = self.env.get_observation()

        # Use CLIPort to predict action
        action = self._cliport.predict(obs, text)

        # Execute the predicted action
        obs, reward, done, info = self.env.step({
            'pick': action['pick'],
            'place': action['place']
        })

        info['text_instruction'] = text
        info['cliport_action'] = action

        return obs, reward, done, info

    def render(self):
        """Render the environment."""
        if len(self.cached_video_frames) > 0:
            return self.cached_video_frames.pop(0)
        return self.env.get_camera_image()

    def get_observation(self):
        """Get current observation without stepping."""
        return self.env.get_observation()


# Create the environment instance for SHARPIE runner
environment = EnvironmentWrapper()