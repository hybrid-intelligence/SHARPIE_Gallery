# Simple tag

Simple tag experiment, this requires 2 simultaneous users. The adversary needs to chase the agent.

## Installation

From the gallery root:
```bash
python install.py tag
```

## Dependencies

- pettingzoo[mpe]

## Configuration

This use case has the following agents:

- **Fleeing agent** (agent_0): human inputs
  - Keyboard controls:
    - ← (Left)
    - → (Right)
    - ↓ (Down)
    - ↑ (Up)
- **Adversary agent** (adversary_0): human inputs
  - Keyboard controls:
    - ← (Left)
    - → (Right)
    - ↓ (Down)
    - ↑ (Up)

See `config.yaml` for full configuration details.