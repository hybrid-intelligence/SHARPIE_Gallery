# Fetch

Interact with a robotic agent trying to catch a ball. You can move the ball around with keyboard inputs


## Installation

From the gallery root:
```bash
sharpie-install fetch --gallery-dir .
```

### ⚠️ Important

Requires egl back-end library. Install with `sudo apt install libglew-dev`

## Dependencies

- apets-ariel[fetch] @ git+https://github.com/kgd-al/apets-ariel
- git+https://github.com/kgd-al/ariel

## Configuration

This use case has the following agents:

- **Ball** (fetch_agent): human inputs
  - Keyboard controls:
    - ← (Left)
    - ↑ (Up)
    - → (Right)
    - ↓ (Down)

See `config.yaml` for full configuration details.