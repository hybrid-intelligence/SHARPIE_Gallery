"""
SMACv2 Environment Wrapper for SHARPIE.

Provides a clean interface for SMACv2 environments with RGB screen capture.
"""

import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional

from .rgb_capture import (
    RGBStarCraftCapabilityEnvWrapper,
    SMACV2_AVAILABLE,
)


# Capability configuration for terran_5_vs_5 scenario
TERRAN_5v5_CONFIG = {
    "n_units": 5,
    "n_enemies": 5,
    "team_gen": {
        "dist_type": "weighted_teams",
        "unit_types": ["marine", "marauder", "medivac"],
        "weights": [0.45, 0.45, 0.1],
        "exception_unit_types": ["medivac"],
        "observe": True
    },
    "start_positions": {
        "dist_type": "surrounded_and_reflect",
        "p": 0.5,
        "map_x": 32,
        "map_y": 32
    }
}


class EnvironmentWrapper:
    """
    Wrapper for SMACv2 environment compatible with SHARPIE.

    Uses RGBStarCraftCapabilityEnvWrapper for actual SC2 game graphics.
    """

    def __init__(
        self,
        map_name: str = "10gen_terran",
        capability_config: Optional[Dict] = None,
        max_steps: int = 5000,
        debug: bool = False,
        rgb_width: int = 1920,
        rgb_height: int = 1080,
    ):
        if not SMACV2_AVAILABLE:
            raise ImportError(
                "SMACv2 is not installed. Install it with:\n"
                "pip install git+https://github.com/oxwhirl/smacv2.git"
            )

        self.map_name = map_name
        self.capability_config = capability_config or TERRAN_5v5_CONFIG
        self.max_steps = max_steps
        self.debug = debug
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        # Initialize with RGB capture enabled
        self.env = RGBStarCraftCapabilityEnvWrapper(
            capability_config=self.capability_config,
            map_name=map_name,
            debug=debug,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
        )

        # Get environment info
        self.env_info = self.env.get_env_info()
        self.n_agents = self.env_info["n_agents"]
        self.n_actions = self.env_info["n_actions"]
        self.obs_shape = self.env_info["obs_shape"]
        self.state_shape = self.env_info.get("state_shape", None)
        self.step_count = 0
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.env.reset()
        self.step_count = 0

        obs_list = self.env.get_obs()
        observations = {
            agent_id: obs for agent_id, obs in zip(self.agent_ids, obs_list)
        }
        state = self.env.get_state() if hasattr(self.env, 'get_state') else None

        info = {}
        return observations, info

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        t0 = time.time()
        # Validate actions against available action masks
        validated_actions = []
        for i, agent_id in enumerate(self.agent_ids):
            action = actions.get(agent_id, 0)
            # Get available actions for this agent
            avail_actions = self.env.get_avail_agent_actions(i)
            # If action is not available, pick first available action
            if avail_actions is not None and action < len(avail_actions):
                if avail_actions[action] == 0:
                    # Action not available, find first available
                    available = np.flatnonzero(avail_actions)
                    if len(available) > 0:
                        action = int(available[0])
                    else:
                        action = 0  # NOOP as fallback
            validated_actions.append(action)
        t1 = time.time()

        reward, terminated, info = self.env.step(validated_actions)
        t2 = time.time()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        obs_list = self.env.get_obs()
        t3 = time.time()

        observations = {
            agent_id: obs for agent_id, obs in zip(self.agent_ids, obs_list)
        }
        state = self.env.get_state() if hasattr(self.env, 'get_state') else None

        info = {}

        if self.debug:
            print(f"step timing: action_validate={t1-t0:.3f}s, env.step={t2-t1:.3f}s, get_obs={t3-t2:.3f}s")

        return observations, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Render the environment - returns actual SC2 game graphics.

        Returns:
            RGB array of shape (height, width, 3). Never returns None.
        """
        t0 = time.time()
        try:
            # Try to get RGB from SC2 observation
            rgb = self.env._rgb_env.get_rgb_screen()
            t1 = time.time()
            if self.debug:
                print(f"render timing: get_rgb_screen={t1-t0:.3f}s")
            if rgb is not None and rgb.size > 0:
                return rgb

            # Skip pygame render as it can block in headless environments
            # Return a placeholder image instead
            return self._create_placeholder()
        except Exception as e:
            if self.debug:
                print(f"Warning: Rendering failed: {e}")
            return self._create_placeholder()

    def _create_placeholder(self) -> np.ndarray:
        """Create a placeholder image when rendering fails."""
        img = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
        # Add visual indicator that rendering failed
        cv2.putText(img, "Render Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def get_avail_agent_actions(self, agent_id: str) -> np.ndarray:
        if agent_id in self.agent_ids:
            agent_idx = self.agent_ids.index(agent_id)
            return self.env.get_avail_agent_actions(agent_idx)
        raise ValueError(f"Unknown agent_id: {agent_id}")

    def get_obs(self) -> List[np.ndarray]:
        return self.env.get_obs()

    def get_state(self) -> Optional[np.ndarray]:
        return self.env.get_state() if hasattr(self.env, 'get_state') else None

    def get_env_info(self) -> Dict[str, Any]:
        return self.env_info

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


# Create default environment instance
environment = EnvironmentWrapper(debug=True)