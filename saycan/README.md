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

Requires Python 3.10, Ollama (https://ollama.ai), and gdown (`pip install gdown`). Assets (robot URDFs, ViLD model, CLIPort checkpoint, ~1GB) are automatically downloaded from Google Drive on first environment initialization.

## Dependencies

- ftfy
- regex
- tqdm
- fvcore
- git+https://github.com/openai/CLIP.git
- gdown
- moviepy
- imageio
- imageio-ffmpeg
- opencv-python
- pillow
- matplotlib
- ipython
- pybullet
- ollama
- easydict
- tensorflow
- torch
- torchvision
- jax[cuda]
- flax
- optax
- numpy
- scipy

## Configuration

This use case has the following agents:

- **Robot** (agent_0): human inputs (policy: SayCan)

See `config.yaml` for full configuration details.