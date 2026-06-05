# SMACv2

SMACv2 (StarCraft Multi-Agent Challenge v2) is a benchmark for cooperative multi-agent reinforcement learning based on Blizzard's StarCraft II. Control Terran units in cooperative 5v5 battles against enemy forces.

## Installation

From the gallery root:
```bash
python install.py smacv2
```

### ⚠️ Important

Requires StarCraft II installation and SMAC_Maps. See [SMACv2 GitHub](https://github.com/oxwhirl/smacv2) for detailed installation instructions.

## Dependencies

- git+https://github.com/oxwhirl/smacv2.git
- numpy
- opencv-python-headless

## Configuration

This use case has the following agents:

- **Unit 1** (agent_0): human inputs (policy: SMACv2 Heuristic)
  - Keyboard controls:
    - ↑ (North)
    - ↓ (South)
    - ← (West)
    - → (East)

- **Unit 2-5** (agent_1-4): AI-controlled (policy: SMACv2 Heuristic)

See `config.yaml` for full configuration details.