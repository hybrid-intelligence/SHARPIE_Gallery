# SHARPIE Gallery

Shared Human-AI Reinforcement Learning Platform for Interactive Experiments Gallery

This repository presents use-cases for [SHARPIE](https://github.com/hybrid-intelligence/SHARPIE).

## Quick Start

### Prerequisites
Activate your Python environment (conda or venv) with SHARPIE installed:
```bash
# Using conda
conda activate sharpie

# Or using venv
source /path/to/venv/bin/activate

# Ensure the database is set up and that there is an admin (superuser)
sharpie-web migrate
sharpie-web createsuperuser
```

### Install all use cases
```bash
# navigate to SHARPIE root directory
python install.py --all --connection-key secret
```

### Install specific use case
```bash
python install.py amaze                    # Standard output
python install.py amaze --quiet             # Minimal output (errors only)
python install.py amaze --verbose           # Detailed output
```

### Custom installation paths
The default install location is the install location for the ``sharpie`` package. If ``sharpie`` has not been installed, the install script looks at `../SHARPIE` (relative to the gallery directory). As a third option, custom paths for the root sharpie-dir and the webserver-dir can be given. Both arguments are optional and can be used independently:
```bash
python install.py amaze --sharpie-dir /path/to/SHARPIE           # webserver defaults to <sharpie-dir>/webserver
python install.py amaze --webserver-dir /path/to/webserver       # sharpie-dir defaults to ../SHARPIE
python install.py amaze --sharpie-dir /path/to/SHARPIE --webserver-dir /path/to/webserver
```

### List available use cases
```bash
python install.py --list
```

### Validate without installing
```bash
python install.py amaze --check            # Standard output
python install.py amaze --check --quiet    # Minimal output
python install.py amaze --check --verbose  # Detailed output
python install.py --all --check            # Validate all
```

## Available Use Cases

- **amaze** - Maze navigation with TAMER
- **frozen** - Frozen lake with TAMER  
- **mario** - Super Mario Bros behavior cloning
- **mountain** - Mountain car (human-only)
- **mountain_tamer** - Mountain car with TAMER
- **overcooked** - Collaborative cooking
- **saycan** - Language-conditioned robotic manipulation
- **smacv2** - StarCraft II multi-agent challenge
- **spread** - Multi-agent coordination
- **tag** - Multi-agent pursuit

## Interacting with installed use case
Open a terminal window, activate your environment:
```bash
# Using conda
conda activate sharpie

# Or using venv
source /path/to/venv/bin/activate
sharpie-web runserver
```
Simultaneously open a second terminal window and run:
```bash
# Using conda
conda activate sharpie

# Or using venv
source /path/to/venv/bin/activate
sharpie-runner runserver --connection-key secret
```
Open a browser and visit [localhost:8000](localhost:8000) to see all installed experiments.

## Development

### Testing
Run validation on all use cases:
```bash
python validate_all.py
```

### Adding a New Use Case

1. Create directory: `my_use_case/`
2. Add `config.yaml` (see existing examples for schema)
3. Add `environment.py` (must define `environment` variable)
4. Add `policy.py` if needed (must define `policy` variable with class named `Policy`)
5. Validate: `python install.py my_use_case --check`
6. Install: `python install.py my_use_case`

### Regenerating READMEs
READMEs are generated from config.yaml files:
```bash
python generate_readme.py
```

## Configuration Schema

Each use case's `config.yaml` contains:
- `use_case` - Unique identifier
- `python_version` - **(Optional)** Required Python version (defaults to '3.13' if not specified)
- `dependencies` - List of pip packages to install
- `installation_notes` - **(Optional)** Installation notes (e.g., "Requires Python >= 3.13")
- `environment` - Environment configuration (name, description, filepaths)
- `policy` - **(Optional)** Policy configuration
- `agents` - List of agent configurations
- `experiment` - Experiment configuration

### Environment Configuration

The `environment` section defines the simulation environment:
- `name` - Display name for the environment
- `description` - Description of what the environment does
- `filepaths` - Dictionary containing:
  - `environment` - Path to environment.py file (required)

### Policy Configuration

The optional `policy` section defines the agent policy:
- `name` - Policy name
- `description` - **(Optional)** Policy description
- `filepaths` - Dictionary containing:
  - `policy` - Path to policy.py file (required)
  - Additional support files (e.g., `tamer`, `human_expert`, `rgb_capture`) (optional)
- `checkpoint_interval` - How often to save model checkpoints (0 = disabled)

### Agent Configuration Fields

Each agent in the `agents` list can include:
- `role` - Agent identifier (e.g., agent_0, agent_1)
- `name` - Display name
- `description` - Agent description
- `policy` - Policy name (or null for human-only control)
- `participant` - Whether agent participates in experiments
- `keyboard_inputs` - Key mapping for actions
- `keyboard_input_display` - **(Optional)** Display configuration for keyboard inputs
- `multiple_keyboard_inputs` - **(Optional)** Allow multiple simultaneous key presses (default: false)
- `inputs_type` - **(Optional)** Input type: "actions", "reward", or "other" (default: "other")
- `textual_inputs` - **(Optional)** Accept text input (default: false)

### Experiment Configuration

The `experiment` section defines how the experiment runs:
- `link` - URL slug/identifier for the experiment
- `name` - Display name
- `short_description` - Brief description for listings
- `long_description` - Detailed description with instructions
- `enabled` - Whether the experiment is active (boolean)
- `number_of_episodes` - Number of episodes to run
- `target_fps` - Target frames per second for rendering
- `wait_for_inputs` - Whether to pause for human input each step (boolean)

## Python Version Compatibility

Different use cases require different Python versions. The `python_version` field in `config.yaml` specifies the required version:

| Use Case | Python Version | Notes |
|----------|---------------|-------|
| overcooked | 3.10.x only | overcooked_ai requires `>=3.10,<3.11` |
| saycan | 3.10.x only | JAX and TensorFlow compatibility |
| smacv2 | 3.10.x only | SMAC dependencies |
| mario | >= 3.13 | gym-super-mario-bros, nes-py |
| amaze | >= 3.13 | Default |
| frozen | >= 3.13 | Default |
| mountain | >= 3.13 | Default |
| mountain_tamer | >= 3.13 | Default |
| spread | >= 3.13 | Default |
| tag | >= 3.13 | Default |

The CI pipeline tests each use case on its required Python version.