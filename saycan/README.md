# SayCan - Language-Conditioned Robotic Manipulation

This experiment implements the SayCan approach for grounding language in robotic affordances,
combining:
- **ViLD**: Open-vocabulary object detection
- **LLM (Ollama)**: Task planning and action scoring
- **CLIPort**: Language-conditioned pick-and-place manipulation
- **PyBullet**: Physics simulation with UR5e robot arm

## Installation

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Ollama (for LLM)
Install Ollama from [ollama.ai](https://ollama.ai) and pull a model:
```bash
ollama pull llama3.2:1b
```

### Asset Downloads
Assets (robot URDFs, ViLD model, CLIPort checkpoint) are downloaded automatically on first run.

## Setting on the Webserver

<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='SayCan', defaults={
    'description': 'Language-conditioned robotic manipulation with LLM planning',
    'filepaths': {'environment': 'saycan/environment.py'}
})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='saycan', defaults={
    'name': 'SayCan',
    'short_description': 'Robot manipulation with natural language instructions',
    'long_description': 'Guide a robot arm to pick and place objects using natural language instructions. The system uses ViLD for object detection, an LLM for task planning, and CLIPort for language-conditioned manipulation.\r\n\n<br>\n<br>\nYou can give instructions like:\n<ul>\n<li>\"task: put all blocks in bowls\" - Set a high-level task</li>\n<li>\"pick the blue block and place it on the red bowl\" - Direct instruction</li>\n</ul>',
    'enabled': True,
    'environment': Environment.objects.get(name='SayCan'),
    'number_of_episodes': 1,
    'target_fps': 24.0,
    'wait_for_inputs': True
})"
```

<!-- Add this policy to the database -->
```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='SayCan', defaults={
    'description': 'SayCan policy with LLM planning and CLIPort execution',
    'filepaths': {'policy': 'saycan/policy.py'},
    'checkpoint_interval': 0
})"
```

<!-- Add this agent to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent, Policy; Agent.objects.update_or_create(role='agent_0', defaults={
    'name': 'Robot',
    'description': 'UR5e robot arm with Robotiq gripper',
    'policy': Policy.objects.get(name='SayCan'),
    'participant': True,
    'keyboard_inputs': {},
    'multiple_keyboard_inputs': False,
    'inputs_type': 'actions',
    'textual_inputs': True
})"
```

<!-- Link agent to experiment -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='saycan'); exp.agents.add(Agent.objects.get(role='agent_0'))"
```

## Usage

### Action Types
The environment accepts the following action types:

| Action | Description |
|--------|-------------|
| `"task:<description>"` | Set a high-level task for LLM planning |
| `"plan"` | Get next planned action from LLM |
| `"<text instruction>"` | Direct pick-and-place instruction |
| `"done"` | End the episode |

### Example Tasks
- `task: put all blocks in bowls`
- `task: stack the blocks`
- `task: sort blocks by color`
- `pick the blue block and place it on the red bowl`

## References

- **SayCan**: [Ahn et al. (2022) - Do As I Can, Not As I Say](https://arxiv.org/abs/2204.01691)
- **CLIPort**: [Shridhar et al. (2021) - What and Where Pathways for Robotic Manipulation](https://arxiv.org/abs/2109.12098)
- **ViLD**: [Gu et al. (2021) - Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921)

## Repository

- Original SayCan: https://github.com/google-research/google-research/tree/master/saycan
- CLIPort: https://github.com/cliport/cliport