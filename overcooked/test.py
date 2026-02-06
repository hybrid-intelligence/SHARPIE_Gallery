"""
Had to pip install
    'tensorflow==2.11' 'ray[rllib,tune]==2.0' dm-tree 'gym<0.24' 'numpy<2' 'opencv-python-headless<4.13' gitpython 'pydantic<2'

Failed attempt at reading from existing RllibAsymmetricAdvantagesSP (from overcooked_demo)
"""

import os
import pickle
from pathlib import Path

import dill
import gymnasium
import overcooked_demo
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, AgentFromPolicy
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from human_aware_rl.rllib.rllib import gen_trainer_from_params
from ray.rllib.algorithms.ppo import PPO

from environment import FixedOvercooked


AGENT_DIR = Path(overcooked_demo.__file__).parent.joinpath(f"server/static/assets/agents/")


def get_policy(npc_id, idx=0):
    if npc_id.lower().startswith("rllib"):
        # Loading rllib agents requires additional helpers
        fpath = os.path.join(AGENT_DIR, npc_id, "agent")
        fix_bc_path(fpath)
        agent = load_agent(fpath, agent_index=idx)
        return agent
    else:
        try:
            fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
            with open(fpath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError("Error loading agent\n{}".format(e.__repr__()))


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)



def load_trainer(save_path, true_num_workers=False):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint directory
    Additionally we decide if we want to use the same number of remote workers (see ray library Training APIs)
    as we store in the previous configuration, by default = False, we use only the local worker (see ray library API)
    """
    # Read in params used to create trainer
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    if not true_num_workers:
        # Override this param to lower overhead in trainer creation
        config["training_params"]["num_workers"] = 0

    if config["training_params"]["num_gpus"] == 1:
        # all other configs for the server can be kept for local testing
        config["training_params"]["num_gpus"] = 0

    if "trained_example" in save_path:
        # For the unit testing we update the result directory in order to avoid an error
        config["results_dir"] = (
            "/Users/runner/work/human_aware_rl/human_aware_rl/human_aware_rl/ppo/results_temp"
        )

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)
    #
    # # Search for the checkpoint folder in the save_path that has the highest number. Checkpoint folders are named checkpoint_<number>
    # checkpoint_dirs = [d for d in os.listdir(save_path) if d.startswith("checkpoint_")]
    # if not checkpoint_dirs:
    #     raise ValueError(f"No checkpoint directories found in {save_path}")
    #
    # # Find directory with highest checkpoint number (2025: this was recently changed because previously checkpoints didn't have this format)
    # latest_dir = checkpoint_dirs[0]
    # latest_num = int(latest_dir.split("_")[1].lstrip("0") or "0")
    # for d in checkpoint_dirs[1:]:
    #     num = int(d.split("_")[1].lstrip("0") or "0")
    #     if num > latest_num:
    #         latest_num = num
    #         latest_dir = d
    #
    # checkpoint_file = os.path.join(save_path, latest_dir + "/")

    checkpoints = list(Path(save_path).glob("**/checkpoint*"))
    checkpoint_file = sorted(checkpoints)[-1]

    # Load weights into dummy object
    trainer.restore(checkpoint_file)
    return trainer


def get_agent_from_trainer(trainer, policy_id="ppo", agent_index=0):
    policy = trainer.get_policy(policy_id)
    dummy_env = trainer.env_creator(trainer.config["env_config"])
    featurize_fn = dummy_env.featurize_fn_map[policy_id]
    agent = RlLibAgent(policy, agent_index, featurize_fn=featurize_fn)
    return agent


def load_agent(save_path, policy_id="ppo", agent_index=0):
    """
    Returns an RllibAgent (compatible with the Overcooked Agent API) from the `save_path` to a previously
    serialized trainer object created with `save_trainer`

    The trainer can have multiple independent policies, so extract the one with ID `policy_id` to wrap in
    an RllibAgent

    Agent index indicates whether the agent is player zero or player one (or player n in the general case)
    as the featurization is not symmetric for both players
    """
    trainer = load_trainer(save_path)
    return get_agent_from_trainer(trainer, policy_id=policy_id, agent_index=agent_index)


if __name__ == '__main__':
    print("Hello")

    room = "asymmetric_advantages"
    agent_type = "SP"

    folder_name = "".join(token.capitalize() for token in room.split("_"))
    agent_path = f"Rllib{folder_name}{agent_type}"

    config_path = AGENT_DIR.joinpath(agent_path).joinpath("config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
        print("==========")
        print("= Config =")
        print(config)
        print("==========")

    print("===========")
    print("= Trainer =")
    print("(init)")
    trainer: PPO = gen_trainer_from_params(config)
    print(trainer)
    print("===========")
    trainer.load_checkpoint(AGENT_DIR.joinpath(agent_path).joinpath("checkpoint-650"))
    print("(loading)")

    print(trainer)
    print("===========")

    # with open("../overcooked_ai/src/overcooked_demo/server/static/assets/agents/RllibAsymmetricAdvantagesSP/config.pkl", "rb") as f:
    #     print(pickle.load(f))
    # PPO().load_checkpoint("../overcooked_ai/src/overcooked_demo/server/static/assets/agents/RllibAsymmetricAdvantagesSP/agent/checkpoint-650")
    # with open("../overcooked_ai/src/overcooked_demo/server/static/assets/agents/RllibAsymmetricAdvantagesSP/agent/checkpoint-650", "rb") as f:
    #     print(pickle.load(f))

    mdp = OvercookedGridworld.from_layout_name(room)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=500)
    # env = OvercookedPZ(base_env=base_env, agents=AgentPair(RandomAgent(), RandomAgent()))

    env = gymnasium.make("Overcooked-v1", base_env=base_env, featurize_fn=base_env.featurize_state_mdp, render_mode="human")
    print(env)

    agents = [RandomAgent(env.action_space), AgentFromPolicy(get_policy(agent_path))]

    obs, info = env.reset()

    done = False
    while not done:
        # obs, reward, done, info = env.step(tuple(a.action(obs) for a in agents))
        obs, reward, done, info = env.step(tuple(env.action_space.sample() for a in agents))
        env.render()
    print()
