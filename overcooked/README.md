# Overcooked

Serve as many orders as possible with an AI teammate

## Installation

From the gallery root:
```bash
python install.py overcooked
```

### ⚠️ Important

Requires Python >= 3.10, < 3.11

## Dependencies

- git+https://github.com/HumanCompatibleAI/overcooked_ai.git

## Configuration

This use case has the following agents:

- **Agent** (overcooked_agent): AI inputs (policy: OvercookedRandom)
- **Human** (overcooked_human): human inputs
  - Keyboard controls:
    - ↑ (Up)
    - ↓ (Down)
    - ← (Left)
    - → (Right)
    - ↵ (Interact)

See `config.yaml` for full configuration details.