# SMACv2 Terran 5v5

Guide your units in a cooperative 5v5 battle against enemy forces. Use arrow keys to influence movement while AI heuristics handle combat decisions.

Controls:
- ↑ - Move North
- ↓ - Move South
- ← - Move West
- → - Move East

Features randomized unit compositions (Marines, Marauders, Medivacs) and start positions.

## Installation

From the gallery root:
```bash
python install.py smacv2
```

### ⚠️ Important

Requires StarCraft II installation and SMAC_Maps. See https://github.com/oxwhirl/smacv2 for installation instructions.

## Dependencies

- git+https://github.com/oxwhirl/smacv2.git
- numpy
- opencv-python-headless

## Configuration

This use case has the following agents:

- **Unit 1** (smacv2_agent_0): human inputs (policy: SMACv2 Heuristic)
  - Keyboard controls:
    - ↑ (North)
    - ↓ (South)
    - ← (West)
    - → (East)
- **Unit 2** (smacv2_agent_1): AI inputs (policy: SMACv2 Heuristic)
- **Unit 3** (smacv2_agent_2): AI inputs (policy: SMACv2 Heuristic)
- **Unit 4** (smacv2_agent_3): AI inputs (policy: SMACv2 Heuristic)
- **Unit 5** (smacv2_agent_4): AI inputs (policy: SMACv2 Heuristic)

See `config.yaml` for full configuration details.