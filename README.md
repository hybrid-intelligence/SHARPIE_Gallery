# SHARPIE Gallery

Shared Human-AI Reinforcement Learning Platform for Interactive Experiments Gallery

This repository presents use-cases for [SHARPIE](https://github.com/hybrid-intelligence/SHARPIE).

## Quick Start

### Install all use cases
```bash
python install.py --all
```

### Install specific use case
```bash
python install.py amaze
```

### List available use cases
```bash
python install.py --list
```

### Validate without installing
```bash
python install.py amaze --check
python install.py --all --check  # Validate all
```

## Available Use Cases

- **amaze** - Maze navigation with TAMER
- **frozen** - Frozen lake with TAMER  
- **mountain** - Mountain car (human-only)
- **mountain_tamer** - Mountain car with TAMER
- **overcooked** - Collaborative cooking
- **spread** - Multi-agent coordination
- **tag** - Multi-agent pursuit

## Development

### Testing
Run validation on all use cases:
```bash
python validate_all.py
```

### Adding a New Use Case

1. Create directory: `my_use_case/`
2. Add `config.json` (see existing examples for schema)
3. Add `environment.py` (must define `environment` variable)
4. Add `policy.py` if needed (must define `policy` variable with class named `Policy`)
5. Validate: `python install.py my_use_case --check`
6. Install: `python install.py my_use_case`

### Regenerating READMEs
READMEs are generated from config.json files:
```bash
python generate_readme.py
```

## Configuration Schema

Each use case's `config.json` contains:
- `use_case` - Unique identifier
- `dependencies` - List of pip packages to install
- `environment` - Environment configuration (name, description, filepaths)
- `policy` - Optional policy configuration
- `agents` - List of agent configurations with keyboard_input_display
- `experiment` - Experiment configuration

## Requirements

- Python 3.10+
- SHARPIE (main repository in adjacent directory)
- Django 5.2+