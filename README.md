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

Paper Landing Webpage: https://dfki-ric-underactuated-lab.github.io/orthant_rewards_biped_rl/

# Citation
```bibtex
@inproceedings{HLGKK23,
  author = { Harnack,  Daniel and
             L\"{u}th, Christoph and
             Gross, Lukas and
             Kumar, Shivesh and 
             Kirchner, Frank}
  title =  { Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking},
  booktitle = {62nd {IEEE} Conference on Decision and Control ({CDC})},
  address   = {Marina Bay Sands, Singapore} 
  pages     = {2135 -- 2140},
  year      = {2023},
  publisher = {{IEEE}}
}
```
