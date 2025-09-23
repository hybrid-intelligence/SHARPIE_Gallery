# SHARPIE Gallery
## Shared Human-AI Reinforcement Learning Platform for Interactive Experiments Gallery

This repository presents use-cases for [SHARPIE](https://github.com/libgoncalv/SHARPIE/). 

### Usage instructions
* Install SHARPIE by following the instructions on the main repository
* Choose one of the use-cases and start by installing the requirements `pip install -r requirements.txt`
* Copy the content of the folder `webserver/` to the folder `webserver/experiment/` of the SHARPIE repository
* Copy the content of the folder `runner/` to the folder `runner/` of the SHARPIE repository
* Lauch the webserver and the runner as explained on the SHARPIE repository

### Content of the use-cases
Each use-case folder contains configuration files for the interface and logging (webserver) and the backend environment and agent(s) (runner). This includes:
* Webserver
    * settings.py
        * Configuration form
        * Experiment name
        * Inputs listened in the browser
* Runner
    * settings.py
        * Input mapping from the captured inputs to the action(s) needed by the environment
        * Termination condition of the environment
        * Number of human users per experiment
    * environment.py
        * environment instance, this can be any environment as long as it defines reset(), step(action) and render()
    * agent.py
        * agent instances list, this can be any function as long as it defines sample(obs)