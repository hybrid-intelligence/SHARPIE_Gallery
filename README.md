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
- **mountain** - Mountain car (human-only)
- **mountain_tamer** - Mountain car with TAMER
- **overcooked** - Collaborative cooking
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
- `policy` - Optional policy configuration
- `agents` - List of agent configurations with keyboard_input_display
- `experiment` - Experiment configuration

## Requirements

- Python 3.10+ (varies by use case)
- SHARPIE (main repository in adjacent directory)
- Django 5.2+

## Python Version Compatibility

Different use cases may require different Python versions. The `python_version` field in `config.yaml` specifies the required version for each use case:

| Use Case | Python Version | Reason |
|----------|---------------|--------|
| overcooked | 3.10.x only | Dependency: overcooked_ai requires `>=3.10,<3.11` |
| mario | >= 3.13 | Dependencies: gym-super-mario-bros>=8.0.0, nes-py>=9.0.0 |
| all others | >= 3.13 (default) | Compatible with Python 3.13+ |

The CI pipeline tests each use case on its required Python version.