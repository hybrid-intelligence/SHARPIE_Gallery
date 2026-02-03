from pathlib import Path

AGENT_DIR = Path(__file__).parent


def get_policy(self, npc_id, idx=0):
    if npc_id.lower().startswith("rllib"):
        try:
            # Loading rllib agents requires additional helpers
            fpath = os.path.join(AGENT_DIR, npc_id, "agent")
            fix_bc_path(fpath)
            agent = load_agent(fpath, agent_index=idx)
            return agent
        except Exception as e:
            raise IOError(
                "Error loading Rllib Agent\n{}".format(e.__repr__())
            )
    else:
        try:
            fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
            with open(fpath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError("Error loading agent\n{}".format(e.__repr__()))


if __name__ == '__main__':
    print("Hello")

