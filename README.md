# SHARPIE Gallery
## Shared Human-AI Reinforcement Learning Platform for Interactive Experiments Gallery

This repository presents use-cases for [SHARPIE](https://github.com/libgoncalv/SHARPIE/). 

### Usage instructions
* Install SHARPIE by following the instructions on the main repository
* Choose one (or several) of the use-cases and follow the instructions in the README.md file
* Copy the experiments you selected under the `experiments` folder of the runner
* Lauch the webserver as explained on the SHARPIE repository and the runner using `cd runner && python manage.py runserver`.
Relaunch the runner if it was already running.

### Content of the use-cases
Each use-case folder contains configuration files for the backend environment and agent(s). This includes:
* environment.py
    * Environment instance, this can be any environment as long as it defines reset(), step(action) and render()
    * Input mapping from the captured inputs to the action(s) needed by the environment
    * Termination condition of the environment
* agent.py
    * Agent instances list, this can be any class as long as it defines sample(obs)
