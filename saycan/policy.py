"""
SayCan Policy for SHARPIE.

A simple policy wrapper for the SayCan environment. The actual LLM planning
and CLIPort execution are handled by the environment module.

Original SayCan Repository:
    https://github.com/google-research/google-research/tree/master/saycan

Reference:
    Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in
    Robotic Affordances. arXiv preprint arXiv:2204.01691.
"""


class Policy:
    """
    SayCan-based policy for pick-and-place operations.

    This policy passes participant inputs directly to the environment,
    which handles LLM planning and CLIPort execution.
    """

    def __init__(self, room_name=""):
        """
        Initialize the SayCan policy.

        Args:
            room_name: Optional room identifier (unused, kept for compatibility)
        """
        self.name = "SayCan_Policy"
        self.room_name = room_name

    def predict(self, observation, participant_input=None):
        """
        Predict an action based on the observation.

        Args:
            observation: Current observation from the environment
            participant_input: Text instruction from participant:
                              - "task:<description>" to set task and auto-plan
                              - "plan" to get next planned action
                              - Direct text instruction for CLIPort

        Returns:
            The participant_input (passed through to environment)
        """
        return participant_input

    def update(self, observation, action, reward, done, next_observation):
        """
        Update the policy based on experience (no-op for SayCan).

        SayCan doesn't use traditional RL updates. This method is kept
        for compatibility with the SHARPIE framework.
        """
        pass


# Create an instance of the policy for use by the runner
policy = Policy('saycan')