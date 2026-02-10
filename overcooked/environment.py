import gymnasium
import numpy as np

import overcooked_ai_py.mdp.overcooked_env
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked as OriginalOvercooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action, Direction


_input_mapping = dict(
    ArrowUp=Direction.NORTH,
    ArrowDown=Direction.SOUTH,
    ArrowRight=Direction.EAST,
    ArrowLeft=Direction.WEST,
    Space=Action.STAY,
    Enter=Action.INTERACT,
)


def input_mapping(inputs):
    print(f"[kgd-debug|input_mapping|start] {inputs=}")

    actions = {}
    for agent, keys in inputs.items():
        if len(keys) == 0:
            action = Action.STAY
        else:
            action = _input_mapping[keys[0]]
        print(f"[kgd-debug|input_mapping]    {keys=} -> {action=}")
        actions[agent] = Action.ACTION_TO_INDEX[action]

    print(f"[kgd-debug|input_mapping|end] {inputs=}")
    return actions


def termination_condition(terminated, truncated):
    return terminated or truncated


# -------------------#---
# -- Light wrapper --#---
# -------------------#---


class FixedOvercooked(OriginalOvercooked):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, *args, render_mode=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_mode = "rgb_array"

    def reset(self, seed: 'int | None' = None, options: 'dict[str, Any] | None' = None):
        # Just to follow gym API
        obs = super().reset()
        return tuple(o.astype(np.float32) for o in obs["both_agent_obs"]), obs

    def step(self, actions: dict):  # Convert OvercookedState to dictionary for serialization
        print(f"[kgd-debug|FixedOvercooked.step] {actions=}")
        obs, reward, done, info = super().step(actions.values())
        obs["overcooked_state"] = obs["overcooked_state"].to_dict()
        return obs, reward, done, False, info

    def render(self):  # Convert BGR -> RGB
        return super().render()[:, :, ::-1]


# -------------------#---

from gymnasium.envs.registration import register

register(
    id="Overcooked-v1",
    entry_point="environment:FixedOvercooked",
)


mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
environment = gymnasium.make("Overcooked-v1", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
