import gymnasium as gym

def input_mapping(inputs):
    if len(inputs) == 0:
        return 0
    
    for agent, actions in inputs.items():
        if 'ArrowUp' in actions:
            inputs[agent] = 1
        elif 'ArrowDown' in actions:
            inputs[agent] = -1
        else:
            inputs[agent] = 0

    return inputs['agent_0']

def termination_condition(terminated, truncated):
    return terminated or truncated

environment = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")