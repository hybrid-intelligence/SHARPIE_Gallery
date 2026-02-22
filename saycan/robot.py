"""
Robotiq 2F-85 Gripper Control Module.

This module provides control for the Robotiq 2F-85 parallel gripper in PyBullet
simulation. The gripper is commonly used with UR5e robot arm in manipulation tasks.

Key Features:
- Gripper open/close control
- Grasp detection
- Thread-based constraint enforcement

Original SayCan Repository:
    https://github.com/google-research/google-research/tree/master/saycan

Reference:
    Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in
    Robotic Affordances. arXiv preprint arXiv:2204.01691.
"""

import os
import threading
import time
import numpy as np
import pybullet

# Get the saycan directory for asset paths
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))


class Robotiq2F85:
    """
    Gripper handling for Robotiq 2F-85.

    This class manages the gripper attached to a robot arm, providing
    open/close functionality and grasp detection.

    Attributes:
        robot: PyBullet robot body ID
        tool: Link ID of the robot's tool (end effector)
        body: PyBullet gripper body ID
        n_joints: Number of joints in the gripper
        activated: Whether the gripper is currently activated (grasping)
    """

    def __init__(self, robot, tool):
        """
        Initialize the gripper.

        Args:
            robot: PyBullet body ID of the robot
            tool: Link ID of the robot's end effector
        """
        self.robot = robot
        self.tool = tool
        pos = [0.1339999999999999, -0.49199999999872496, 0.5]
        rot = pybullet.getQuaternionFromEuler([np.pi, 0, np.pi])
        urdf = os.path.join(SAYCAN_DIR, "robotiq_2f_85", "robotiq_2f_85.urdf")
        self.body = pybullet.loadURDF(urdf, pos, rot)
        self.n_joints = pybullet.getNumJoints(self.body)
        self.activated = False

        # Connect gripper base to robot tool
        pybullet.createConstraint(
            self.robot, tool, self.body, 0,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, -0.07],
            childFrameOrientation=pybullet.getQuaternionFromEuler([0, 0, np.pi / 2])
        )

        # Set friction coefficients for gripper fingers
        for i in range(pybullet.getNumJoints(self.body)):
            pybullet.changeDynamics(
                self.body, i,
                lateralFriction=10.0,
                spinningFriction=1.0,
                rollingFriction=1.0,
                frictionAnchor=True
            )

        # Start thread to handle additional gripper constraints
        self.motor_joint = 1
        self.constraints_thread = threading.Thread(target=self.step)
        self.constraints_thread.daemon = True
        self.constraints_thread.start()

    def step(self):
        """Control joint positions by enforcing hard constraints on gripper behavior."""
        while True:
            try:
                currj = [pybullet.getJointState(self.body, i)[0] for i in range(self.n_joints)]
                indj = [6, 3, 8, 5, 10]
                targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
                pybullet.setJointMotorControlArray(
                    self.body, indj, pybullet.POSITION_CONTROL, targj,
                    positionGains=np.ones(5)
                )
            except:
                return
            time.sleep(0.001)

    def activate(self):
        """Activate the gripper (close fingers to grasp)."""
        pybullet.setJointMotorControl2(
            self.body, self.motor_joint,
            pybullet.VELOCITY_CONTROL,
            targetVelocity=1,
            force=10
        )
        self.activated = True

    def release(self):
        """Release the gripper (open fingers)."""
        pybullet.setJointMotorControl2(
            self.body, self.motor_joint,
            pybullet.VELOCITY_CONTROL,
            targetVelocity=-1,
            force=10
        )
        self.activated = False

    def detect_contact(self):
        obj, _, ray_frac = self.check_proximity()
        if self.activated:
            empty = self.grasp_width() < 0.01
            cbody = self.body if empty else obj
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)
        #   else:
        #     return ray_frac < 0.14 or self.external_contact()

    # Return if body is in contact with something other than gripper
    def external_contact(self, body=None):
        if body is None:
            body = self.body
        pts = pybullet.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]
        return len(pts) > 0  # pylint: disable=g-explicit-length-test

    def check_grasp(self):
        while self.moving():
            time.sleep(0.001)
        success = self.grasp_width() > 0.01
        return success

    def grasp_width(self):
        lpad = np.array(pybullet.getLinkState(self.body, 4)[0])
        rpad = np.array(pybullet.getLinkState(self.body, 9)[0])
        dist = np.linalg.norm(lpad - rpad) - 0.047813
        return dist

    def check_proximity(self):
        ee_pos = np.array(pybullet.getLinkState(self.robot, self.tool)[0])
        tool_pos = np.array(pybullet.getLinkState(self.body, 0)[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = pybullet.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac