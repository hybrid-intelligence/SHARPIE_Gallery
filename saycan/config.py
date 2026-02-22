"""
SayCan Configuration and Asset Downloader.

This module provides global configuration constants for the SayCan environment
and handles downloading required assets (robot URDFs, model weights).

Original SayCan Repository:
    https://github.com/google-research/google-research/tree/master/saycan

Reference:
    Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, Y., David, B.,
    Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Ho, D.,
    Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jang, E., Jang, R., Julian, R.,
    ... & Zeng, A. (2022). Do As I Can, Not As I Say: Grounding Language in
    Robotic Affordances. arXiv preprint arXiv:2204.01691.
"""

import collections
import datetime
import os
import random
import threading
import time

import cv2  # Used by ViLD.
import clip
from easydict import EasyDict
import flax
from flax import linen as nn
from flax.training import checkpoints
from flax.metrics import tensorboard
import imageio
from heapq import nlargest
import IPython
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
import numpy as np
import optax
import pickle
from PIL import Image
import pybullet
import pybullet_data
import tensorflow.compat.v1 as tf
import torch
from tqdm import tqdm

import subprocess

# Get the directory where this script is located
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))


def download_assets():
    """Download PyBullet robot assets, ViLD model weights, and CLIPort checkpoint."""
    # Change to saycan directory for downloads
    original_dir = os.getcwd()
    os.chdir(SAYCAN_DIR)

    try:
        # Download PyBullet assets (UR5e robot, Robotiq gripper, bowl)
        if not os.path.exists('ur5e/ur5e.urdf'):
            print("Downloading UR5e robot assets...")
            subprocess.run(['gdown', '--id', '1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc'], check=True)
            subprocess.run(['gdown', '--id', '1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX'], check=True)
            subprocess.run(['gdown', '--id', '1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM'], check=True)

            print("Extracting assets...")
            subprocess.run(['unzip', '-o', 'ur5e.zip'], check=True)
            subprocess.run(['unzip', '-o', 'robotiq_2f_85.zip'], check=True)
            subprocess.run(['unzip', '-o', 'bowl.zip'], check=True)

        # Download ViLD pretrained model weights
        if not os.path.exists('image_path_v2'):
            print("Downloading ViLD model weights...")
            # Try using wget with public URL since gsutil may not be available
            os.makedirs('image_path_v2/variables', exist_ok=True)
            base_url = 'https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2/'
            subprocess.run(['wget', '-q', base_url + 'saved_model.pb', '-O', 'image_path_v2/saved_model.pb'], check=False)
            subprocess.run(['wget', '-q', base_url + 'variables/variables.data-00000-of-00001', '-O', 'image_path_v2/variables/variables.data-00000-of-00001'], check=False)
            subprocess.run(['wget', '-q', base_url + 'variables/variables.index', '-O', 'image_path_v2/variables/variables.index'], check=False)

        # Download CLIPort pretrained checkpoint
        if not os.path.exists('cliport_checkpoint'):
            print("Downloading CLIPort pretrained checkpoint...")
            os.makedirs('cliport_checkpoint', exist_ok=True)
            # CLIPort checkpoint from original SayCan paper
            subprocess.run(['gdown', '--id', '1NqJDTyxZOOqvCM2RZthJT5qPX3Xi-a-g', '-O', 'cliport_checkpoint/checkpoint'], check=False)

        # Download training dataset (optional, for fine-tuning)
        if not os.path.exists('dataset-9999.pkl'):
            print("Downloading CLIPort training dataset...")
            subprocess.run(['gdown', '--id', '1yCz6C-6eLWb4SFYKdkM-wz5tlMjbG2h8'], check=False)
    finally:
        os.chdir(original_dir)

# Call download_assets() only when this script is run directly, not when imported
if __name__ == "__main__":
    download_assets()

# =============================================================================
# Global Constants
# =============================================================================

# Objects that can be picked up
PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
}

# RGBA colors for objects
COLORS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (255/255,  87/255,  89/255, 255/255),
    "green":  (89/255,  169/255,  79/255, 255/255),
    "yellow": (237/255, 201/255,  72/255, 255/255),
}

# Target locations for placing objects (None = dynamic, tuple = fixed position)
PLACE_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,

  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,

  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}

# Workspace configuration
PIXEL_SIZE = 0.00267857  # Meters per pixel
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X, Y, Z bounds in meters