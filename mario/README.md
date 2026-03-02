# Installation
Create a virtual name named `mario_env` with Python 3.8 using the following command:

```bash
conda create -n "mario_env" python=3.8
```
After activation of this environment, install necessary packages:

```bash
pip install -r requirements.txt
```

# Setting on the webserver
<!-- Add this environment to the database -->
```bash
python manage.py shell -c "from experiment.models import Environment; Environment.objects.update_or_create(name='Mario', defaults={'description':'Gym Super Mario Bros','filepaths':{'environment':'mario/environment.py'}})"
```

<!-- Add this experiment to the database -->
```bash
python manage.py shell -c "from experiment.models import Experiment, Environment; Experiment.objects.update_or_create(link='mario', defaults={
    'name': 'Mario',
    'conda_environment': 'mario_env',
    'short_description': 'Mario experiment - Single player',
    'long_description': 'Mario experiment - Single player',
    'enabled': True,
    'environment': Environment.objects.get(name='Mario'),
    'number_of_episodes': 1,
    'target_fps': 60.0,
    'wait_for_inputs': False
})"
```

<!-- Add this policy to the database -->

```bash
python manage.py shell -c "from experiment.models import Policy; Policy.objects.update_or_create(name='MarioExpertPolicy', defaults={
    'description': '',
    'filepaths': {'policy': 'mario/policy.py', 'human_expert': 'mario/tamer.py'},
    'checkpoint_interval': 1
})"
```

<!-- Add this agent to the database -->
```bash
python manage.py shell -c "from experiment.models import Agent; Agent.objects.update_or_create(role='agent_1', defaults={
    'name': 'Mario',
    'description': 'Mario Agent to train expert policy',
    'policy': Policy.objects.get(name='MarioExpertPolicy'),
    'participant': True,
    'keyboard_inputs': {'ArrowLeft': 64, 'ArrowRight': 128, 'ArrowUp': 1, 'ArrowDown': 2,'default': 0},
    'multiple_keyboard_inputs': True,
    'inputs_type': 'other',
    'textual_inputs': False
})"
```

<!-- Link agent to experiment -->

```bash
python manage.py shell -c "from experiment.models import Experiment, Agent; exp = Experiment.objects.get(link='mario'); exp.agents.add(Agent.objects.get(role='agent_1'))"
```