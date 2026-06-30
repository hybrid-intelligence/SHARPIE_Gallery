# SayCan

Guide a robot arm to pick and place objects using natural language instructions. The system uses ViLD for object detection, an LLM for task planning, and CLIPort for language-conditioned manipulation.

You can give instructions like:
- "task: put all blocks in bowls" - Set a high-level task
- "pick the blue block and place it on the red bowl" - Direct instruction

## Installation

From the gallery root:
```bash
python install.py saycan
```

### ⚠️ Important

Requires Python 3.10, Ollama (https://ollama.ai), and gdown (pip install gdown). Assets (robot URDFs, ViLD model, CLIPort checkpoint) are automatically downloaded from Google Drive on first environment initialization (~1GB).

## Dependencies

- numpy
- scipy
- torch
- torchvision
- jax[cuda]
- flax
- optax
- tensorflow
- tensorboard
- opencv-python
- pillow
- matplotlib
- imageio
- imageio-ffmpeg
- moviepy
- ftfy
- regex
- tqdm
- fvcore
- git+https://github.com/openai/CLIP.git
- pybullet
- gdown
- easydict
- ollama
- ipython

## Configuration

This use case has the following agents:

- **Robot** (saycan_agent_0): human inputs (policy: SayCan)

See `config.yaml` for full configuration details.