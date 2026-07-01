import os
os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np

from pathlib import Path

from aapets.common.robot_storage import RerunnableRobot
from aapets.common.mujoco.state import MjState
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.zoo.evolve import Arguments as ZooArguments

from aapets.fetch.types import Config as Arguments
from aapets.fetch.dynamics.demo_ball import DemoBallDynamics
from aapets.fetch.sm_fetcher import FetcherCPG
from aapets.fetch.types import InteractionMode


import os
os.system('clear')


class EnvironmentWrapper:
    """Wrapper for the MountainCar-v0 Gymnasium environment."""

    fps = 25.0

    def __init__(self):
        self.args = Arguments()
        self.args.duration = np.inf
        self.args.ball_strength = 25
        self.args.robot_archive = Path(__file__).parent.joinpath("spider.zip")
        self.args.pretty_print()
        self.record = RerunnableRobot.load(self.args.robot_archive)

        self.dynamics_class = DemoBallDynamics
        self.dynamics_class.adjust_world(self.record.mj_spec, self.args)

        if self.args.mode is InteractionMode.HUMAN:
            self.args.camera = f"ortho-cam"
        else:
            self.args.camera = f"{self.args.robot_name_prefix}1_tracking-cam"
        if self.args.camera is not None:  # Adjust camera *before* compilation
            self.dynamics_class.adjust_camera(self.record.mj_spec, self.args)

        self.state = MjState.from_spec(self.record.mj_spec)
        self.model, self.data = self.state.model, self.state.data
        mujoco.mj_forward(self.model, self.data)

        robot = f"{self.args.robot_name_prefix}1_world"

        self.sub_steps = int(1 / (self.model.opt.timestep * self.fps))

        self.viewer = mujoco.Renderer(self.model, width=480, height=480)
        # match self.args.camera:
        #     case None:
        #         pass
        #
        #     case "tracking":
        #         self.viewer.cam.type = mjtCamera.mjCAMERA_TRACKING
        #         self.viewer.cam.trackbodyid = self.model.body(robot).id
        #
        #     case _:
        #         self.viewer.cam.fixedcamid = self.model.camera(self.args.camera).id
        #         self.viewer.cam.type = mjtCamera.mjCAMERA_FIXED

        self.brain = FetcherCPG(
            self.record.brain[-1], **self.record.brain[1],
            state=self.state, name=robot)

        self.dynamics = self.dynamics_class(
            self.state,
            overlay=None,
            robot=robot, ball="ball", human="None",
            brain=self.brain,
            config=self.args,
        )
        self.callbacks = MjcbCallbacks(
            self.state, [self.brain], dict(dynamics=self.dynamics), self.args)
        self.callbacks.start()

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial observation (numpy array)
            info: Additional information
        """
        self.callbacks.stop()
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.brain.reset(self.state)
        self.callbacks.start()
        return self._observation(), self._infos()

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
        actions = list(action_dict.values())[0]
        try:
            actions.remove(0)  # Ignore default action
        except:
            pass
        self.dynamics.set_keys(actions)
        mujoco.mj_step(self.model, self.data, nstep=self.sub_steps)
        return self._observation(), 0, False, False, self._infos()

    def render(self):
        """
        Render the environment.

        Returns:
            image: Rendered image of the environment (numpy array)
        """
        self.viewer.update_scene(self.data, camera=self.args.camera)
        frame = self.viewer.render()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        label = f"t = {self.data.time:.3g} s"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        return frame

    def _observation(self): return np.array([])
    def _infos(self): return dict()


environment = EnvironmentWrapper()
