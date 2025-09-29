# Installation
Run `pip install 'pettingzoo[mpe]'`

# Setting on the webserver
Add on the interface a new experiment with the following information:
* Name `Simple tag`
* Description `Simple tag experiment, this requires 2 simultaneous users. The adversary needs to chase the agent.`
* Inputs listened `["ArrowLeft", "ArrowRight", "ArrowDown", "ArrowUp"]`
* Agent available to play `[["agent_0", "Agent"], ["adversary_0", "Adversary"]]`
* Number of users `2`
* Link `tag`