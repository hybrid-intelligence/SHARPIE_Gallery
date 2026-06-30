# Fetch

Control stuff in the fetch task


## Installation

From the gallery root:
```bash
python install.py fetch
```

### ⚠️ Important

Requires egl back-end library. Install with `sudo apt install libglew-dev`

## Dependencies

- apets-ariel[fetch] @ git+https://github.com/kgd-al/apets-ariel
- git+https://github.com/kgd-al/ariel

## Configuration

This use case has the following agents:

- **Fetcher** (fetch_agent): human inputs
  - Keyboard controls:
    - ← (Left)
    - ↑ (Up)
    - → (Right)
    - ↓ (Down)

See `config.yaml` for full configuration details.