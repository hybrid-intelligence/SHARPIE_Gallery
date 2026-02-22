"""
CLIPort - CLIP + Transporter Networks for Language-Conditioned Manipulation.

This module implements the CLIPort architecture for language-conditioned pick-and-place
operations. It combines CLIP (Contrastive Language-Image Pre-training) with Transporter
Networks to predict pick and place positions based on natural language instructions.

Key Components:
- ResNet-based encoder-decoder architecture
- CLIP text and image encoders
- Transporter Networks for pick and place heatmap prediction
- Pretrained checkpoint loading

CLIPort Repository:
    https://github.com/cliport/cliport

Reference:
    Shridhar, M., Manuelli, L., & Fox, D. (2021). CLIPort: What and Where Pathways
    for Robotic Manipulation. Conference on Robot Learning (CoRL).

Used in SayCan:
    https://github.com/google-research/google-research/tree/master/saycan
"""

import os
import subprocess
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from flax.training import checkpoints
from moviepy import ImageSequenceClip
from IPython.display import display

# Get the saycan directory for checkpoint paths
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))

class ResNetBlock(nn.Module):
  """ResNet pre-Activation block. https://arxiv.org/pdf/1603.05027.pdf"""
  features: int
  stride: int = 1

  def setup(self):
    self.conv0 = nn.Conv(self.features // 4, (1, 1), (self.stride, self.stride))
    self.conv1 = nn.Conv(self.features // 4, (3, 3))
    self.conv2 = nn.Conv(self.features, (1, 1))
    self.conv3 = nn.Conv(self.features, (1, 1), (self.stride, self.stride))

  def __call__(self, x):
    y = self.conv0(nn.relu(x))
    y = self.conv1(nn.relu(y))
    y = self.conv2(nn.relu(y))
    if x.shape != y.shape:
      x = self.conv3(nn.relu(x))
    return x + y


class UpSample(nn.Module):
  """Simple 2D 2x bilinear upsample."""

  def __call__(self, x):
    B, H, W, C = x.shape
    new_shape = (B, H * 2, W * 2, C)
    return jax.image.resize(x, new_shape, 'bilinear')


class ResNet(nn.Module):
  """Hourglass 53-layer ResNet with 8-stride."""
  out_dim: int

  def setup(self):
    self.dense0 = nn.Dense(8)

    self.conv0 = nn.Conv(64, (3, 3), (1, 1))
    self.block0 = ResNetBlock(64)
    self.block1 = ResNetBlock(64)
    self.block2 = ResNetBlock(128, stride=2)
    self.block3 = ResNetBlock(128)
    self.block4 = ResNetBlock(256, stride=2)
    self.block5 = ResNetBlock(256)
    self.block6 = ResNetBlock(512, stride=2)
    self.block7 = ResNetBlock(512)

    self.block8 = ResNetBlock(256)
    self.block9 = ResNetBlock(256)
    self.upsample0 = UpSample()
    self.block10 = ResNetBlock(128)
    self.block11 = ResNetBlock(128)
    self.upsample1 = UpSample()
    self.block12 = ResNetBlock(64)
    self.block13 = ResNetBlock(64)
    self.upsample2 = UpSample()
    self.block14 = ResNetBlock(16)
    self.block15 = ResNetBlock(16)
    self.conv1 = nn.Conv(self.out_dim, (3, 3), (1, 1))

  def __call__(self, x, text):

    # # Project and concatenate CLIP features (early fusion).
    # text = self.dense0(text)
    # text = jnp.expand_dims(text, axis=(1, 2))
    # text = jnp.broadcast_to(text, x.shape[:3] + (8,))
    # x = jnp.concatenate((x, text), axis=-1)

    x = self.conv0(x)
    x = self.block0(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)

    # Concatenate CLIP features (mid-fusion).
    text = jnp.expand_dims(text, axis=(1, 2))
    text = jnp.broadcast_to(text, x.shape)
    x = jnp.concatenate((x, text), axis=-1)

    x = self.block8(x)
    x = self.block9(x)
    x = self.upsample0(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.upsample1(x)
    x = self.block12(x)
    x = self.block13(x)
    x = self.upsample2(x)
    x = self.block14(x)
    x = self.block15(x)
    x = self.conv1(x)
    return x


class TransporterNets(nn.Module):
  """TransporterNet with 3 ResNets (translation only)."""

  def setup(self):
    # Picking affordances.
    self.pick_net = ResNet(1)

    # Pick-conditioned placing affordances.
    self.q_net = ResNet(3)  # Query (crop around pick location).
    self.k_net = ResNet(3)  # Key (place features).
    self.crop_size = 64
    self.crop_conv = nn.Conv(features=1, kernel_size=(self.crop_size, self.crop_size), use_bias=False, dtype=jnp.float32, padding='SAME')

  def __call__(self, x, text, p=None, train=True):
    B, H, W, C = x.shape
    pick_out = self.pick_net(x, text)  # (B, H, W, 1)

    # Get key features.
    k = self.k_net(x, text)

    # Add 0-padding before cropping.
    h = self.crop_size // 2
    x_crop = jnp.pad(x, [(0, 0), (h, h), (h, h), (0, 0)], 'maximum')

    # Get query features and convolve them over key features.
    place_out = jnp.zeros((0, H, W, 1), jnp.float32)
    for b in range(B):

      # Get coordinates at center of crop.
      if p is None:
        pick_out_b = pick_out[b, ...]  # (H, W, 1)
        pick_out_b = pick_out_b.flatten()  # (H * W,)
        amax_i = jnp.argmax(pick_out_b)
        v, u = jnp.unravel_index(amax_i, (H, W))
      else:
        v, u = p[b, :]

      # Get query crop.
      x_crop_b = jax.lax.dynamic_slice(x_crop, (b, v, u, 0), (1, self.crop_size, self.crop_size, x_crop.shape[3]))
      # x_crop_b = x_crop[b:b+1, v:(v + self.crop_size), u:(u + self.crop_size), ...]

      # Convolve q (query) across k (key).
      q = self.q_net(x_crop_b, text[b:b+1, :])  # (1, H, W, 3)
      q = jnp.transpose(q, (1, 2, 3, 0))  # (H, W, 3, 1)
      place_out_b = self.crop_conv.apply({'params': {'kernel': q}}, k[b:b+1, ...])  # (1, H, W, 1)
      scale = 1 / (self.crop_size * self.crop_size)  # For higher softmax temperatures.
      place_out_b *= scale
      place_out = jnp.concatenate((place_out, place_out_b), axis=0)

    return pick_out, place_out


def n_params(params):
  return jnp.sum(jnp.int32([n_params(v) if isinstance(v, dict) or isinstance(v, flax.core.frozen_dict.FrozenDict) else np.prod(v.shape) for v in params.values()]))

from flax.training import train_state

class TrainState(train_state.TrainState):
  pass




#@markdown Train your own model, or load a pretrained one.
load_pretrained = True  #@param {type:"boolean"}

# Initialize model weights using dummy tensors.
rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
init_text = jnp.ones((4, 512), jnp.float32)
init_pix = jnp.zeros((4, 2), np.int32)
init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
print(f'Model parameters: {n_params(init_params):,}')

# Define the Optax optimizer
optimizer_tx = optax.adam(learning_rate=1e-4)

# Create an initial TrainState object. This will have step=0.
optim = TrainState.create(apply_fn=TransporterNets().apply,
                          params=init_params,
                          tx=optimizer_tx)

if load_pretrained:
  ckpt_path = os.path.join(SAYCAN_DIR, f'ckpt_{40000}')
  if not os.path.exists(ckpt_path):
    import subprocess
    print("Downloading CLIPort checkpoint...")
    subprocess.run(['gdown', '--id', '1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP', '-O', ckpt_path], check=False)

  try:
    # Attempt to restore directly. This will fail if 'step' is missing in the checkpoint.
    optim = checkpoints.restore_checkpoint(ckpt_path, optim)
    print('Loaded:', ckpt_path)
  except ValueError as e:
    if "Missing field step in state dict" in str(e):
      print("Attempting to load old checkpoint format (missing 'step' field).")
      # Load the raw checkpoint data as a dictionary
      loaded_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)

      if isinstance(loaded_state_dict, dict):
        # Extract parameters, common keys for parameters are 'params' or 'target'
        params_from_ckpt = loaded_state_dict.get('params', loaded_state_dict.get('target', init_params))
        
        # Re-initialize the opt_state using the current optax optimizer with loaded parameters.
        # This means the exact state of the old optimizer might be lost if it was not optax-compatible,
        # but model parameters are preserved.
        new_opt_state = optimizer_tx.init(params_from_ckpt)

        # Create a new TrainState with the loaded parameters, re-initialized opt_state, and step=0
        optim = TrainState(
            step=0, # Default to step 0 if not present in old checkpoint
            params=params_from_ckpt,
            tx=optimizer_tx,
            opt_state=new_opt_state,
            apply_fn=TransporterNets().apply
        )
        print('Successfully migrated and loaded checkpoint (params restored, opt_state re-initialized, step set to 0).')
      else:
        print(f"Error: Checkpoint '{ckpt_path}' is not a dictionary. Cannot migrate. Using initial model state.")
    else:
      # Re-raise other ValueErrors
      raise

else:

  # Training loop.
  batch_size = 8
  for train_iter in range(1, 40001):
    batch_i = np.random.randint(dataset_size, size=batch_size)
    text_feat = data_text_feats[batch_i, ...]
    img = dataset['image'][batch_i, ...] / 255
    img = np.concatenate((img, np.broadcast_to(coords[None, ...], (batch_size,) + coords.shape)), axis=3)

    # Get onehot label maps.
    pick_yx = np.zeros((batch_size, 2), dtype=np.int32)
    pick_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
    place_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
    for i in range(len(batch_i)):
      pick_y, pick_x  = dataset['pick_yx'][batch_i[i], :]
      place_y, place_x = dataset['place_yx'][batch_i[i], :]
      pick_onehot[i, pick_y, pick_x] = 1
      place_onehot[i, place_y, place_x] = 1
      # pick_onehot[i, ...] = scipy.ndimage.gaussian_filter(pick_onehot[i, ...], sigma=3)

      # Data augmentation (random translation).
      roll_y, roll_x = np.random.randint(-112, 112, size=2)
      img[i, ...] = np.roll(img[i, ...], roll_y, axis=0)
      img[i, ...] = np.roll(img[i, ...], roll_x, axis=1)
      pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_y, axis=0)
      pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_x, axis=1)
      place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_y, axis=0)
      place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_x, axis=1)
      pick_yx[i, 0] = pick_y + roll_y
      pick_yx[i, 1] = pick_x + roll_x

    # Backpropagate.
    batch = {}
    batch['img'] = jnp.float32(img)
    batch['text'] = jnp.float32(text_feat)
    batch['pick_yx'] = jnp.int32(pick_yx)
    batch['pick_onehot'] = jnp.float32(pick_onehot)
    batch['place_onehot'] = jnp.float32(place_onehot)
    rng, batch['rng'] = jax.random.split(rng)
    optim, loss, _, _ = train_step(optim, batch)
    writer.scalar('train/loss', loss, train_iter)

    if train_iter % np.power(10, min(4, np.floor(np.log10(train_iter)))) == 0:
      print(f'Train Step: {train_iter} Loss: {loss}')

    if train_iter % 1000 == 0:
      checkpoints.save_checkpoint('.', optim, train_iter, prefix='ckpt_', keep=100000, overwrite=True)



user_input = 'Pick the yellow block and place it on the blue bowl.'  #@param {type:"string"}

# Show camera image before pick and place.

def run_cliport(obs, text, env=None, clip_model=None, coords=None, optim=None, eval_step_fn=None):
  before = env.get_camera_image()
  prev_obs = obs['image'].copy()

  # Tokenize text and get CLIP features.
  text_tokens = clip.tokenize(text).cuda()
  with torch.no_grad():
    text_feats = clip_model.encode_text(text_tokens).float()
  text_feats /= text_feats.norm(dim=-1, keepdim=True)
  text_feats = np.float32(text_feats.cpu())

  # Normalize image and add batch dimension.
  img = obs['image'][None, ...] / 255
  img = np.concatenate((img, coords[None, ...]), axis=3)

  # Run Transporter Nets to get pick and place heatmaps.
  batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
  pick_map, place_map = eval_step_fn(optim, batch)
  pick_map, place_map = np.float32(pick_map), np.float32(place_map)

  # Get pick position.
  pick_max = np.argmax(np.float32(pick_map)).squeeze()
  pick_yx = (pick_max // 224, pick_max % 224)
  pick_yx = np.clip(pick_yx, 20, 204)
  pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

  # Get place position.
  place_max = np.argmax(np.float32(place_map)).squeeze()
  place_yx = (place_max // 224, place_max % 224)
  place_yx = np.clip(place_yx, 20, 204)
  place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

  # Step environment.
  act = {'pick': pick_xyz, 'place': place_xyz}
  obs, _, _, _ = env.step(act)

  # Show pick and place action.
  plt.title(text)
  plt.imshow(prev_obs)
  plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
  plt.show()

  # Show debug plots.
  plt.subplot(1, 2, 1)
  plt.title('Pick Heatmap')
  plt.imshow(pick_map.reshape(224, 224))
  plt.subplot(1, 2, 2)
  plt.title('Place Heatmap')
  plt.imshow(place_map.reshape(224, 224))
  plt.show()

  # Show video of environment rollout.
  debug_clip = ImageSequenceClip(env.cache_video, fps=25)
  display(debug_clip.ipython_display(autoplay=1, loop=1, center=False))
  env.cache_video = []

  # Show camera image after pick and place.
  plt.subplot(1, 2, 1)
  plt.title('Before')
  plt.imshow(before)
  plt.subplot(1, 2, 2)
  plt.title('After')
  after = env.get_camera_image()
  plt.imshow(after)
  plt.show()

  # return pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx
  return obs


# ============================================================================
# CLIPort Interface Class for easy integration
# ============================================================================

import os

# Get the saycan directory for checkpoint paths
SAYCAN_DIR = os.path.dirname(os.path.abspath(__file__))


class CLIPort:
    """CLIPort model interface for text-conditioned pick and place.

    This class provides a simple interface for using CLIPort without
    needing to manage all the global variables and initialization.
    """

    def __init__(self):
        self.clip_model = None
        self.optim = None
        self.coords = None
        self._initialized = False

    def _init(self):
        """Lazy initialization of CLIP and Transporter models."""
        if self._initialized:
            return

        print("Initializing CLIPort...")

        # Initialize CLIP
        self.clip_model, _ = clip.load("ViT-B/32")
        if torch.cuda.is_available():
            self.clip_model = self.clip_model.cuda()
        self.clip_model.eval()

        # Create coordinate tensor
        h, w = 224, 224
        y_coords = np.linspace(0, 1, h)
        x_coords = np.linspace(0, 1, w)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.coords = np.stack([xx, yy], axis=-1).astype(np.float32)

        # Initialize Transporter Net
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        init_img = jnp.ones((1, 224, 224, 5), jnp.float32)
        init_text = jnp.ones((1, 512), jnp.float32)
        init_params = TransporterNets().init(key, init_img, init_text)['params']

        # Create optimizer state
        optimizer_tx = optax.adam(learning_rate=1e-4)
        self.optim = TrainState.create(
            apply_fn=TransporterNets().apply,
            params=init_params,
            tx=optimizer_tx
        )

        # Try to load checkpoint
        ckpt_path = os.path.join(SAYCAN_DIR, 'ckpt_40000')
        if os.path.exists(ckpt_path):
            try:
                # Attempt to restore directly
                self.optim = checkpoints.restore_checkpoint(ckpt_path, self.optim)
                print(f"Loaded CLIPort checkpoint from {ckpt_path}")
            except ValueError as e:
                if "Missing field step" in str(e):
                    print("Migrating old checkpoint format...")
                    try:
                        # Load the raw checkpoint data as a dictionary
                        loaded_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)
                        if isinstance(loaded_state_dict, dict):
                            # Extract parameters
                            params_from_ckpt = loaded_state_dict.get('params', loaded_state_dict.get('target', init_params))
                            # Re-initialize the opt_state with loaded parameters
                            new_opt_state = optimizer_tx.init(params_from_ckpt)
                            # Create a new TrainState with the loaded parameters
                            self.optim = TrainState(
                                step=0,
                                params=params_from_ckpt,
                                tx=optimizer_tx,
                                opt_state=new_opt_state,
                                apply_fn=TransporterNets().apply
                            )
                            print(f"Successfully migrated checkpoint from {ckpt_path}")
                        else:
                            print(f"Could not migrate checkpoint, using random initialization")
                    except Exception as e2:
                        print(f"Could not load CLIPort checkpoint: {e2}")
                        print("Using random initialization - model may not perform well without training.")
                else:
                    raise
            except Exception as e:
                print(f"Could not load CLIPort checkpoint: {e}")
                print("Using random initialization - model may not perform well without training.")
        else:
            print("No CLIPort checkpoint found, using random initialization.")
            print("Run: python config.py  to download pretrained weights")

        self._initialized = True

    def encode_text(self, text):
        """Encode text instruction using CLIP."""
        with torch.no_grad():
            tokens = clip.tokenize([text])
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            text_feats = self.clip_model.encode_text(tokens).float()
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_feats = text_feats.cpu().numpy()
        return text_feats.astype(np.float32)

    def predict(self, observation, text):
        """
        Predict pick and place coordinates from text instruction.

        Args:
            observation: Dict with 'image' and 'xyzmap'
            text: Text instruction string

        Returns:
            action: Dict with 'pick' and 'place' 3D coordinates
        """
        self._init()

        # Get image and encode text
        image = observation['image']
        xyzmap = observation['xyzmap']
        text_feats = self.encode_text(text)

        # Prepare image batch
        img = image[np.newaxis, ...] / 255.0
        img = np.concatenate([img, self.coords[np.newaxis, ...]], axis=-1)

        # Run inference
        def eval_step(optim, batch):
            pick_out, place_out = TransporterNets().apply(
                {'params': optim.params}, batch['img'], batch['text']
            )
            return pick_out, place_out

        batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
        pick_map, place_map = eval_step(self.optim, batch)
        pick_map, place_map = np.float32(pick_map[0]), np.float32(place_map[0])

        # Get pick position
        pick_max = np.argmax(pick_map.flatten())
        pick_y, pick_x = np.unravel_index(pick_max, (224, 224))
        pick_y, pick_x = np.clip(pick_y, 20, 204), np.clip(pick_x, 20, 204)
        pick_xyz = xyzmap[pick_y, pick_x]

        # Get place position
        place_max = np.argmax(place_map.flatten())
        place_y, place_x = np.unravel_index(place_max, (224, 224))
        place_y, place_x = np.clip(place_y, 20, 204), np.clip(place_x, 20, 204)
        place_xyz = xyzmap[place_y, place_x]

        return {
            'pick': pick_xyz,
            'place': place_xyz,
            'pick_map': pick_map,
            'place_map': place_map
        }


# Global CLIPort instance
_cliport = None


def get_cliport():
    """Get or create the global CLIPort instance."""
    global _cliport
    if _cliport is None:
        _cliport = CLIPort()
    return _cliport