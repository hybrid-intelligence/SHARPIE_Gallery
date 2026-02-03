# Installation
Do not run `pip install overcooked-ai`.

Do use [git sources](https://github.com/HumanCompatibleAI/overcooked_ai?tab=readme-ov-file#building-from-source-)

Requires python 3.10

# Setting on the webserver
Add on the interface a new experiment with the following information:
* Name `Overcooked-AI`
* Description `An experiment where both human and AI are part of the environment`
* Inputs listened `["ArrowUp", "ArrowDown"]`
* Agent available to play `[["agent_0", "Agent"]]`
* Number of users `1`
* Link `overcooked`
* Train `False`
* 