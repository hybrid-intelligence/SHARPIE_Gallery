import gymnasium
import numpy as np
import overcooked_ai_py.mdp.overcooked_env
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


def input_mapping(inputs):
    print(f"[kgd-debug|input_mapping|start] {inputs=}")
    if len(inputs) == 0:
        raise RuntimeError("No inputs. Is that possible?")
    
    for agent, actions in inputs.items():
        print(f"[kgd-debug|input_mapping]    {inputs=}")
        if 'ArrowUp' in actions:
            inputs[agent] = 1
        elif 'ArrowDown' in actions:
            inputs[agent] = -1
        else:
            inputs[agent] = 0

    print(f"[kgd-debug|input_mapping|end] {inputs=}")
    return inputs['agent']


def termination_condition(terminated, truncated):
    return terminated or truncated


# --------------#---
# -- Fast-Fix --#---
# --------------#---


OriginalOvercooked = Overcooked


class FixedOvercooked(OriginalOvercooked):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gymnasium.spaces.Tuple([self.observation_space, self.observation_space])
        self.action_space = gymnasium.spaces.Tuple([self.action_space, self.action_space])

    def reset(self, seed: 'int | None' = None, options: 'dict[str, Any] | None' = None):
        obs = super().reset()
        return tuple(o.astype(np.float32) for o in obs["both_agent_obs"]), obs


overcooked_ai_py.mdp.overcooked_env.Overcooked = FixedOvercooked


# -------------#---


mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
environment = gymnasium.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
# environment = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")
