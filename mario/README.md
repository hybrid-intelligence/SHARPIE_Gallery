# Mario

Train an expert policy for Super Mario Bros through behavior cloning from human demonstrations

## Installation

From the gallery root:
```bash
python install.py mario
```

### ⚠️ Important

Requires Python >= 3.14.

## Dependencies

- websockets
- opencv-python-headless
- gymnasium>=1.0.0
- gym-super-mario-bros>=8.0.0
- nes-py>=9.0.0
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