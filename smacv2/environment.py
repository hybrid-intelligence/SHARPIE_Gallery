"""
SMACv2 Environment Wrapper for SHARPIE with RGB Screen Capture.

Enables actual StarCraft II game graphics instead of pygame visualization.
"""

import os
# Set SDL_VIDEODRIVER for headless SC2 rendering
if 'SDL_VIDEODRIVER' not in os.environ:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional

try:
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    SMACV2_AVAILABLE = True
except ImportError:
    SMACV2_AVAILABLE = False
    StarCraft2Env = None
    StarCraftCapabilityEnvWrapper = None


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


class RGBStarCraft2Env(StarCraft2Env):
    """
    Subclass of StarCraft2Env that enables RGB screen capture.

    Overrides _launch() to set want_rgb=True for actual SC2 game graphics.
    """

    def __init__(self, rgb_width: int = 1920, rgb_height: int = 1080, **kwargs):
        """
        Initialize with RGB capture enabled.

        Args:
            rgb_width: Width of RGB screen capture
            rgb_height: Height of RGB screen capture
            **kwargs: Additional arguments passed to StarCraft2Env
        """
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self._obs = None
        super().__init__(**kwargs)

    def _launch(self):
        """
        Override _launch to enable RGB screen capture.
        """
        print("DEBUG RGBStarCraft2Env._launch: Starting SC2 with RGB enabled", flush=True)
        from pysc2 import run_configs
        from pysc2 import maps
        from s2clientprotocol import common_pb2 as sc_common
        from s2clientprotocol import sc2api_pb2 as sc_pb

        # Race and difficulty mappings (same as parent class)
        races = {
            "R": sc_common.Random,
            "P": sc_common.Protoss,
            "T": sc_common.Terran,
            "Z": sc_common.Zerg
        }
        difficulties = {
            "1": sc_pb.VeryEasy,
            "2": sc_pb.Easy,
            "3": sc_pb.Medium,
            "4": sc_pb.MediumHard,
            "5": sc_pb.Hard,
            "6": sc_pb.Harder,
            "7": sc_pb.VeryHard,
            "8": sc_pb.CheatVision,
            "9": sc_pb.CheatMoney,
            "A": sc_pb.CheatInsane
        }

        self._run_config = run_configs.get(version=self.game_version)
        self.version = self._run_config.version
        _map = maps.get(self.map_name)

        # Enable RGB capture in interface options using render spatial setup
        # width controls camera field of view in world units (larger = more zoomed out)
        # For SMAC maps (typically 32-64 world units), use larger width to see full map
        camera_width = 128  # World units - larger values show more of the map
        option = sc_pb.InterfaceOptions(
            raw=True,
            score=False,
            render=sc_pb.SpatialCameraSetup(
                resolution=sc_common.Size2DI(x=self.rgb_width, y=self.rgb_height),
                minimap_resolution=sc_common.Size2DI(x=self.rgb_width, y=self.rgb_height),
                width=camera_width
            )
        )
        print(f"DEBUG _launch: InterfaceOptions render resolution={self.rgb_width}x{self.rgb_height}, width={camera_width}", flush=True)

        # Start SC2 with RGB enabled
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size,
            want_rgb=True
        )
        print("DEBUG _launch: SC2 process started with want_rgb=True", flush=True)

        self._controller = self._sc2_proc.controller

        # Create game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self._bot_race],
            difficulty=difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        # Join game with RGB options
        join = sc_pb.RequestJoinGame(
            race=races[self._agent_race],
            options=option
        )
        self._controller.join_game(join)

        # Rest of initialization (same as parent)
        game_info = self._controller.game_info()
        map_info = game_info.start_raw

        self.map_play_area_min = map_info.playable_area.p0
        self.map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = (
            self.map_play_area_max.x - self.map_play_area_min.x
        )
        self.max_distance_y = (
            self.map_play_area_max.y - self.map_play_area_min.y
        )
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y
        print(f"DEBUG _launch: Map size = {self.map_x}x{self.map_y} world units", flush=True)

        # Initialize pathing grid and terrain height
        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array([
                    [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                    for row in vals
                ], dtype=bool)
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data),
                            dtype=bool
                        ).reshape(self.map_x, self.map_y)
                    ), axis=1
                )
            )

        self.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self.map_x, self.map_y
                    )
                ), 1
            ) / 255
        )

    def get_rgb_screen(self) -> Optional[np.ndarray]:
        """
        Get RGB screen from the most recent SC2 observation.
        Combines map and minimap into a single image.

        Returns:
            RGB array of shape (height, width*2, 3) containing map+minimap side by side
        """
        try:
            # Access the controller's last observation directly
            if hasattr(self, '_controller') and self._controller:
                obs = self._controller.observe()
                if obs and hasattr(obs, 'observation'):
                    render_data = obs.observation.render_data

                    # Get map (main screen)
                    map_img = None
                    if hasattr(render_data, 'map') and render_data.map:
                        data = render_data.map.data
                        height = render_data.map.size.y
                        width = render_data.map.size.x
                        if height > 0 and width > 0 and len(data) > 0:
                            img = np.frombuffer(data, dtype=np.uint8)
                            expected_rgb = height * width * 3
                            expected_rgba = height * width * 4
                            if len(data) == expected_rgb:
                                map_img = img.reshape((height, width, 3))
                            elif len(data) == expected_rgba:
                                map_img = img.reshape((height, width, 4))[:, :, :3]

                    # Get minimap
                    minimap_img = None
                    if hasattr(render_data, 'minimap') and render_data.minimap:
                        data = render_data.minimap.data
                        height = render_data.minimap.size.y
                        width = render_data.minimap.size.x
                        if height > 0 and width > 0 and len(data) > 0:
                            img = np.frombuffer(data, dtype=np.uint8)
                            expected_rgb = height * width * 3
                            expected_rgba = height * width * 4
                            if len(data) == expected_rgb:
                                minimap_img = img.reshape((height, width, 3))
                            elif len(data) == expected_rgba:
                                minimap_img = img.reshape((height, width, 4))[:, :, :3]

                    # Combine map and minimap - minimap as small overlay in corner
                    if map_img is not None and minimap_img is not None:
                        # Resize minimap to be smaller (1/4 size = 256x256 if map is 1024x1024)
                        minimap_small = cv2.resize(
                            minimap_img,
                            (map_img.shape[1] // 4, map_img.shape[0] // 4),
                            interpolation=cv2.INTER_AREA
                        )
                        # Place minimap in bottom-left corner with a border
                        result = map_img.copy()
                        mm_h, mm_w = minimap_small.shape[:2]
                        # Position: bottom-left corner with small margin
                        y_offset = map_img.shape[0] - mm_h - 10
                        x_offset = 10
                        # Draw white border around minimap
                        cv2.rectangle(
                            result,
                            (x_offset - 2, y_offset - 2),
                            (x_offset + mm_w + 1, y_offset + mm_h + 1),
                            (255, 255, 255),
                            2
                        )
                        # Overlay minimap
                        result[y_offset:y_offset + mm_h, x_offset:x_offset + mm_w] = minimap_small
                        print(f"DEBUG get_rgb_screen: map({map_img.shape}) with minimap overlay({minimap_small.shape})", flush=True)
                        return result
                    elif map_img is not None:
                        print(f"DEBUG get_rgb_screen: only map available, shape={map_img.shape}", flush=True)
                        return map_img
                    elif minimap_img is not None:
                        print(f"DEBUG get_rgb_screen: only minimap available, shape={minimap_img.shape}", flush=True)
                        return minimap_img

        except Exception as e:
            print(f"RGB capture failed: {e}")
            import traceback
            traceback.print_exc()
        return None

    def _parse_render_data(self, render_data) -> Optional[np.ndarray]:
        """Parse render data from SC2 observation."""
        try:
            # The render data contains raw bytes
            data = render_data.data
            height = render_data.size.y
            width = render_data.size.x
            data_len = len(data)
            # Check for valid dimensions
            if height <= 0 or width <= 0 or data_len == 0:
                return None
            img = np.frombuffer(data, dtype=np.uint8)
            # Determine if data is RGB (3 channels) or RGBA (4 channels)
            expected_rgb = height * width * 3
            expected_rgba = height * width * 4
            if data_len == expected_rgb:
                return img.reshape((height, width, 3))
            elif data_len == expected_rgba:
                return img.reshape((height, width, 4))[:, :, :3]
            else:
                print(f"DEBUG _parse_render_data: unexpected data size {data_len} for {width}x{height}", flush=True)
                return None
        except Exception as e:
            print(f"DEBUG _parse_render_data error: {e}")
            return None

    def step(self, actions):
        """Override step to capture observation for RGB extraction."""
        result = super().step(actions)
        # Store the observation for RGB extraction
        if hasattr(self, '_controller') and self._controller:
            try:
                self._obs = self._controller.observe()
            except Exception:
                self._obs = None
        return result

    def reset(self, reset_config=None):
        """Override reset to capture observation for RGB extraction."""
        result = super().reset(reset_config)
        # Store the observation for RGB extraction
        if hasattr(self, '_controller') and self._controller:
            try:
                self._obs = self._controller.observe()
            except Exception:
                self._obs = None
        return result


class RGBStarCraftCapabilityEnvWrapper(StarCraftCapabilityEnvWrapper):
    """
    Wrapper that uses RGBStarCraft2Env instead of StarCraft2Env.

    This enables RGB screen capture from SC2.
    """

    def __init__(self, rgb_width: int = 1920, rgb_height: int = 1080, **kwargs):
        """
        Initialize with RGB capture enabled.

        Args:
            rgb_width: Width of RGB screen capture
            rgb_height: Height of RGB screen capture
            **kwargs: Additional arguments passed to parent
        """
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height

        # Store capability_config before parent init
        self.distribution_config = kwargs.get("capability_config", {})
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()

        # Create RGB-enabled env instead of default StarCraft2Env
        self.env = RGBStarCraft2Env(
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            **kwargs
        )

        # Store reference to the RGB-enabled env for easy access
        self._rgb_env = self.env


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

        reward, terminated, info = self.env.step(validated_actions)
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        obs_list = self.env.get_obs()
        observations = {
            agent_id: obs for agent_id, obs in zip(self.agent_ids, obs_list)
        }
        state = self.env.get_state() if hasattr(self.env, 'get_state') else None

        info = {}
        return observations, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Render the environment - returns actual SC2 game graphics.

        Returns:
            RGB array of shape (height, width, 3). Never returns None.
        """
        # Debug: print where render is called from
        import traceback
        print("DEBUG render: called from:", flush=True)
        traceback.print_stack()
        print(flush=True)

        try:
            # Try to get RGB from SC2 observation
            rgb = self.env._rgb_env.get_rgb_screen()
            if rgb is not None and rgb.size > 0:
                print(f"DEBUG render: returning RGB from get_rgb_screen, shape={rgb.shape}")
                return rgb
            print(f"DEBUG render: get_rgb_screen returned empty or None, using placeholder", flush=True)

            # Skip pygame render as it can block in headless environments
            # Return a placeholder image instead
            placeholder = self._create_placeholder()
            print(f"DEBUG render: returning placeholder shape={placeholder.shape}", flush=True)
            return placeholder
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_placeholder()

    def _create_placeholder(self) -> np.ndarray:
        """Create a placeholder image when rendering fails."""
        print("DEBUG _create_placeholder: creating placeholder", flush=True)
        img = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
        # Add visual indicator that rendering failed
        cv2.putText(img, "Render Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print(f"DEBUG _create_placeholder: done, shape={img.shape}", flush=True)
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
environment = EnvironmentWrapper()