# Overcooked

Serve as many orders as possible with an AI teammate

## Installation

From the gallery root:
```bash
python install.py overcooked
```

### ⚠️ Important

Requires Python 3.10.

## Dependencies

- git+https://github.com/HumanCompatibleAI/overcooked_ai.git

## Configuration

This use case has the following agents:

- **Agent** (Agent): AI inputs (policy: OvercookedRandom)
- **Human** (Human): human inputs
  - Keyboard controls:
    - ↑ (Up)
    - ↓ (Down)
    - ← (Left)
    - → (Right)
    - ↵ (Interact)

See `config.yaml` for full configuration details.