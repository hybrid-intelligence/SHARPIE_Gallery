from environment import *
from agent import *

from collections.abc import Iterable

def input_mapping(inputs):
    if len(inputs) == 0:
        return {'agent_0':0, 'adversary_0':0}
    
    for agent, actions in inputs.items():
        if 'ArrowLeft' in actions:
            inputs[agent] = 1
        elif 'ArrowRight' in actions:
            inputs[agent] = 2
        elif 'ArrowDown' in actions:
            inputs[agent] = 3
        elif 'ArrowUp' in actions:
            inputs[agent] = 4
        else:
            inputs[agent] = 0

    return inputs

def termination_condition(terminated, truncated):
    if isinstance(terminated, Iterable):
        return all(a == 0 for a in terminated)
    return terminated

user_per_experiment = 2