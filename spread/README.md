# Installation
Run `pip install 'pettingzoo[mpe]'`

# Setting on the webserver
Add on the interface a new experiment with the following information:
* Name `Simple spread`
* Description `Simple spread experiment. Each agent should cover one target.`
* Inputs listened `["ArrowLeft", "ArrowRight", "ArrowDown", "ArrowUp"]`
* Agent available to play `[["agent_0", "Agent"]]`
* Number of users `1`
* Link `spread`