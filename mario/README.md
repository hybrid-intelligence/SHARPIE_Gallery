# Mario

Train an expert policy for Super Mario Bros through behavior cloning from human demonstrations

## Installation

From the gallery root:
```bash
python install.py mario
```

## Dependencies

- websockets
- opencv-python-headless
- gym==0.24.1
- gym-super-mario-bros
- stable-baselines3
- imitation

## Configuration

This use case has the following agents:

- **Mario** (agent_1): human inputs (policy: MarioExpertPolicy)
  - Keyboard controls:
    - ← (Left)
    - → (Right)
    - ↑ (Speed)
    - ␣ (Jump)

See `config.yaml` for full configuration details.