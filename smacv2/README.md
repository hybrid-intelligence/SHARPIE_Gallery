# SMACv2 Environment for SHARPIE

This module provides integration with [SMACv2](https://github.com/oxwhirl/smacv2) (StarCraft Multi-Agent Challenge v2) for multi-agent reinforcement learning experiments in SHARPIE.

## Overview

SMACv2 is a benchmark for cooperative multi-agent reinforcement learning based on Blizzard's StarCraft II. This environment wrapper allows you to run SMACv2 scenarios within the SHARPIE platform, enabling human-in-the-loop experiments with multi-agent coordination.

### Scenario

The default configuration uses `terran_5_vs_5`:
- 5 friendly Terran units (Marines, Marauders, Medivacs with randomized compositions)
- 5 enemy units with randomized positions
- Goal: Eliminate all enemy units

### Current Limitations

- **Rendering**: RGB capture from SC2 is currently disabled. The environment returns a placeholder image. Full game visualization requires additional SC2 configuration.
- **Participant Setup**: Only one agent should be linked to a participant to avoid blocking. Other agents run autonomously with their assigned policy.

## Installation

### 1. Install SMACv2

```bash
pip install git+https://github.com/oxwhirl/smacv2.git
```

### 2. Install StarCraft II

**Linux:**
```bash
# Download SC2 from Blizzard
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip

# Extract to a location (default expected: ~/StarCraftII/)
unzip SC2.4.10.zip -d ~/

# Set environment variable if installed elsewhere
export SC2PATH=/path/to/StarCraftII
```

**macOS/Windows:**
Download and install from [Blizzard's website](https://starcraft2.com/).

### 3. Download SMAC Maps

Download the `SMAC_Maps` folder from [SMACv2 releases](https://github.com/oxwhirl/smacv2/releases) and place it in your StarCraft II directory:

```bash
# After downloading SMAC_Maps.zip
unzip SMAC_Maps.zip -d ~/StarCraftII/Maps/
```

## Usage

### Testing the Environment

```python
from runner.smacv2.environment import environment

# Reset and get initial observations
obs, info = environment.reset()
print(f"Number of agents: {info['n_agents']}")
print(f"Agent IDs: {info['agent_ids']}")
print(f"Observation shapes: {[o.shape for o in obs.values()]}")

# Take random actions
actions = {f"agent_{i}": 0 for i in range(5)}
obs, reward, terminated, truncated, info = environment.step(actions)
print(f"Reward: {reward}, Terminated: {terminated}")

# Render (returns RGB image or None)
img = environment.render()
if img is not None:
    print(f"Image shape: {img.shape}")
```

### Testing the Policy

```python
from runner.smacv2.policy import policy
from runner.smacv2.environment import environment

obs, info = environment.reset()

# Get action for agent_0
agent_obs = obs["agent_0"]
action = policy.predict(agent_obs)
print(f"Selected action: {action}")

# With participant guidance
action_with_guidance = policy.predict(agent_obs, participant_input=1)
print(f"Action with guidance (move up): {action_with_guidance}")
```

## Configuration

### Environment Parameters

The `EnvironmentWrapper` accepts the following parameters:

```python
EnvironmentWrapper(
    map_name="10gen_terran",      # SMACv2 map/scenario name
    capability_config={...},      # Unit configuration
    max_steps=500,                # Max steps before truncation
    debug=False,                  # Enable debug mode
    render_mode="rgb_array"       # Rendering mode
)
```

### Available Maps

SMACv2 supports various scenarios:
- `10gen_terran` - 10v10 Terran units
- `10gen_protoss` - 10v10 Protoss units
- `10gen_zerg` - 10v10 Zerg units
- And many more (see SMACv2 documentation)

### Capability Configuration

Customize unit composition and positioning:

```python
config = {
    "n_units": 5,
    "n_enemies": 5,
    "team_gen": {
        "dist_type": "weighted_teams",
        "unit_types": ["marine", "marauder", "medivac"],
        "weights": [0.45, 0.45, 0.1],
        "exception_unit_types": ["medivac"],
        "observe": True
    },
    "start_positions": {
        "dist_type": "surrounded_and_reflect",
        "p": 0.5,
        "map_x": 32,
        "map_y": 32
    }
}
```

## Setting on the Webserver

<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='SMACv2 Terran 5v5', defaults={
    'description': 'SMACv2 StarCraft II multi-agent benchmark environment. Control units in cooperative battles against enemy forces with randomized unit compositions and start positions.',
    'filepaths': {'environment': 'smacv2/environment.py'}
})"
```

<!-- Add this policy to the database -->
```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='SMACv2 Heuristic', defaults={
    'description': 'Heuristic policy for SMACv2 with participant guidance. Attack nearest enemy if in range, move towards enemies otherwise.',
    'filepaths': {'policy': 'smacv2/policy.py'},
    'checkpoint_interval': 0
})"
```

<!-- Add agent for agent_0 (participant-controlled) -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_0', defaults={
    'name': 'Unit 1',
    'description': 'Controls unit 1 in the battle. Use arrow keys to guide movement.',
    'policy': Policy.objects.get(name='SMACv2 Heuristic'),
    'participant': True,
    'keyboard_inputs': {'ArrowUp': 1, 'ArrowDown': 2, 'ArrowLeft': 3, 'ArrowRight': 4, 'default': -1},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'other',
    'textual_inputs': False
})"
```

<!-- Add agent for agent_1 (AI-controlled) -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_1', defaults={
    'name': 'Unit 2',
    'description': 'AI-controlled unit using heuristic policy.',
    'policy': Policy.objects.get(name='SMACv2 Heuristic'),
    'participant': False,
    'keyboard_inputs': {},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'other',
    'textual_inputs': False
})"
```

<!-- Add agent for agent_2 (AI-controlled) -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_2', defaults={
    'name': 'Unit 3',
    'description': 'AI-controlled unit using heuristic policy.',
    'policy': Policy.objects.get(name='SMACv2 Heuristic'),
    'participant': False,
    'keyboard_inputs': {},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'other',
    'textual_inputs': False
})"
```

<!-- Add agent for agent_3 (AI-controlled) -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_3', defaults={
    'name': 'Unit 4',
    'description': 'AI-controlled unit using heuristic policy.',
    'policy': Policy.objects.get(name='SMACv2 Heuristic'),
    'participant': False,
    'keyboard_inputs': {},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'other',
    'textual_inputs': False
})"
```

<!-- Add agent for agent_4 (AI-controlled) -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_4', defaults={
    'name': 'Unit 5',
    'description': 'AI-controlled unit using heuristic policy.',
    'policy': Policy.objects.get(name='SMACv2 Heuristic'),
    'participant': False,
    'keyboard_inputs': {},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'other',
    'textual_inputs': False
})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='smacv2', defaults={
    'name': 'SMACv2 Terran 5v5',
    'short_description': 'Control Terran units in a 5v5 StarCraft II battle',
    'long_description': 'Guide your units in a cooperative 5v5 battle against enemy forces. Use arrow keys to influence movement while AI heuristics handle combat decisions.\r\n\n<br>\n<br>\n<b>Controls:</b>\n<ul>\n<li>↑ - Move North</li>\n<li>↓ - Move South</li>\n<li>← - Move West</li>\n<li>→ - Move East</li>\n</ul>\n<br>\nFeatures randomized unit compositions (Marines, Marauders, Medivacs) and start positions.',
    'enabled': True,
    'environment': Environment.objects.get(name='SMACv2 Terran 5v5'),
    'number_of_episodes': 1,
    'target_fps': 10.0,
    'wait_for_inputs': False
})"
```

<!-- Link agents to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='smacv2'); exp.agents.add(Agent.objects.get(role='agent_0'), Agent.objects.get(role='agent_1'), Agent.objects.get(role='agent_2'), Agent.objects.get(role='agent_3'), Agent.objects.get(role='agent_4'))"
```

## Action Space

SMACv2 uses discrete actions:
- `0`: NOOP (no operation)
- `1`: STOP
- `2`: MOVE_NORTH
- `3`: MOVE_SOUTH
- `4`: MOVE_EAST
- `5`: MOVE_WEST
- `6+`: ATTACK_ENEMY_X (attack specific enemy by index)

Note: Action IDs may vary by map. Use `environment.get_avail_agent_actions(agent_id)` to get valid actions.

## Troubleshooting

### SC2PATH not found
```bash
export SC2PATH=/path/to/StarCraftII
```

### Maps not found
Ensure `SMAC_Maps` folder is in the StarCraft II Maps directory.

### Rendering issues
The environment uses a placeholder image for rendering by default, as SC2's RGB capture may return empty dimensions in headless environments. This is handled automatically by the wrapper.

If you see `Configure: render interface disabled` in the SC2 logs, RGB capture from SC2 is not available, but the placeholder image will still be transmitted.

### Runner blocks waiting for participant input
**Important:** Only set `participant: True` for ONE agent in your experiment configuration. If multiple agents have `participant: True`, the runner will block waiting for inputs from all participants before continuing.

For multi-agent experiments, configure the additional agents with `participant: False` and let their policies control them:
```python
# Only agent_0 is participant-controlled
Agent.objects.update_or_create(role='agent_0', defaults={
    'participant': True,  # This one waits for human input
    ...
})

# Other agents are policy-controlled
Agent.objects.update_or_create(role='agent_1', defaults={
    'participant': False,  # No human input required
    ...
})
```

## References

- [SMACv2 GitHub](https://github.com/oxwhirl/smacv2)
- [SMACv2 Paper](https://arxiv.org/abs/2212.07489)
- [StarCraft II](https://starcraft2.com/)