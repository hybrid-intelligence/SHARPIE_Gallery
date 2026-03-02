from environment import EnvironmentWrapper
from human_expert import HumanExpert

class Policy:
    
    def __init__(self, id_=""):
        self.name = "Expert_Policy"
        self.id = id_
        
        self.env = EnvironmentWrapper()
        self.env.reset()

        model_file = f"{id}_model" if id else None
        self.human_expert = HumanExpert(model_file_to_load=model_file)

    def predict(self, observation, participant_input=None):
        self.env.step({"dummy":participant_input})      # to simulate and save actual transition with next_obs
        return participant_input

    def update(self, observation, action, reward, done, next_observation):
        action = self.env.translate_action(action)
        next_observation, _, _, info = self.env.last_transition     # overwrite the dummy next_observation with actual one
        enum_action = list(self.env.action_meanings.keys()).index(action)   # get enumerated action for policy
        self.human_expert.add_transition(observation, enum_action, next_observation, reward, done, info)

        if done:
            self.human_expert.train()
            self.human_expert.save_policy(f"{self.id}_policy")


policy = Policy('save')
