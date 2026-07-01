# Frozen lake

Navigate a frozen lake grid using keyboard feedback to train a TAMER agent.

## Installation

From the gallery root:
```bash
sharpie-install frozen --gallery-dir .
```

## Dependencies

- gymnasium
- scikit-learn

## Configuration

This use case has the following agents:

- **Frozen agent** (frozen_agent): human inputs (policy: Frozen)
  - Keyboard controls:
    - ↑ (Good)
    - ↓ (Bad)

See `config.yaml` for full configuration details.