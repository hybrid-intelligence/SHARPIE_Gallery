"""
RGB Screen Capture for SMACv2.

Enables actual StarCraft II game graphics instead of pygame visualization.
"""

import os
# Set SDL_VIDEODRIVER for headless SC2 rendering
if 'SDL_VIDEODRIVER' not in os.environ:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np
from typing import Optional

try:
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
    from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
    SMACV2_AVAILABLE = True
except ImportError:
    SMACV2_AVAILABLE = False
    StarCraft2Env = None
    StarCraftCapabilityEnvWrapper = None


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

        # Start SC2 with RGB enabled
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size,
            want_rgb=True
        )

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
        Parent class already stores observation in self._obs after step()/reset().

        Returns:
            RGB array of shape (height, width, 3)
        """
        try:
            if self._obs is not None and hasattr(self._obs, 'observation'):
                render_data = self._obs.observation.render_data

                # Get map (main screen) - prefer over minimap
                if hasattr(render_data, 'map') and render_data.map:
                    data = render_data.map.data
                    height = render_data.map.size.y
                    width = render_data.map.size.x
                    if height > 0 and width > 0 and len(data) > 0:
                        img = np.frombuffer(data, dtype=np.uint8)
                        expected_rgb = height * width * 3
                        expected_rgba = height * width * 4
                        if len(data) == expected_rgb:
                            return img.reshape((height, width, 3))
                        elif len(data) == expected_rgba:
                            return img.reshape((height, width, 4))[:, :, :3]

                # Fallback to minimap if map not available
                if hasattr(render_data, 'minimap') and render_data.minimap:
                    data = render_data.minimap.data
                    height = render_data.minimap.size.y
                    width = render_data.minimap.size.x
                    if height > 0 and width > 0 and len(data) > 0:
                        img = np.frombuffer(data, dtype=np.uint8)
                        expected_rgb = height * width * 3
                        expected_rgba = height * width * 4
                        if len(data) == expected_rgb:
                            return img.reshape((height, width, 3))
                        elif len(data) == expected_rgba:
                            return img.reshape((height, width, 4))[:, :, :3]

        except Exception:
            pass
        return None


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