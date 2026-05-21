# Simple spread

Simple spread experiment. Each agent should cover one target.

## Installation

From the gallery root:
```bash
python install.py spread
```

## Dependencies

- pettingzoo[mpe]

## Configuration

This use case has the following agents:

- **Agent 1** (agent_0): human inputs (policy: Spread agent)
  - Keyboard controls:
    - ← (Left)
    - → (Right)
    - ↓ (Down)
    - ↑ (Up)
- **Agent 2** (agent_1): AI inputs (policy: Spread agent)
- **Agent 3** (agent_2): AI inputs (policy: Spread agent)

See `config.yaml` for full configuration details.