# Orthant Walking RL

The main entry points for training and running the trained agents are the
scripts `scripts/train_rl_agent.py` and `scripts/run_trained_agent.py`
The script `scripts/example_virtual_gravity.py` contains the
reference implementation with the virtual gravity controller.

Training, model, and simulation are controlled by the
respective json files in `parameters`. The most interesting 
is probably rl_parameters.json, through which the reward setup
can be controlled. If you want to add another reward term 
with the name xxx, implement the function in 
`rl_environments/compass_walker_env.py` as CompassWalkerEnv._reward_xxx.
Setting "xxx": weight under "reward_setup" in `parameters/rl_parameters.json` 
will then use this reward in the final sum of reward terms. 

Trained agents are saved automatically under `results/trained_agents/datetime` 
along with a copy of the used parameters for reproducibility.
