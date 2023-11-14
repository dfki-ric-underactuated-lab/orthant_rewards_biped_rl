# Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking

This repository contains the sources for the work described in the paper _Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking_ by Harnack _et al_, to be presented at the  [62nd IEEE Conference on Decision and Control](https://www.cdc2023.ieeecss.org), Dec. 13-15, 2023, Singapore.

## Abstract

Generating physical movement behaviours from their symbolic description is a long-standing challenge in artificial intelligence (AI) and robotics, requiring insights into numerical optimization methods as well as into formalizations from symbolic AI and reasoning. In this paper, a novel approach to finding a reward function from a symbolic description is proposed. The intended system behaviour is modelled as a hybrid automaton, which reduces the system state space to allow more efficient reinforcement learning. The approach is applied to bipedal walking, by modelling the walking robot as a hybrid automaton over state space orthants, and used with the compass walker to derive a reward that incentivizes following the hybrid automaton cycle. As a result, training times of reinforcement learning controllers are reduced while final walking speed is increased. The approach can serve as a blueprint how to generate reward functions from symbolic AI and reasoning. 

## Video

You can watch the video showing the learned behaviours [here](https://youtu.be/CkvLvz_tLtc).

## Citation

You can provisionally cite the paper as 
```
@inproceedings{HLGKK23,
  author = { Harnack,  Daniel and
             L\"{u}th, Christoph and
             Gross, Lukas and
             Kumar, Shivesh and 
             Kirchner, Frank Kirchner}
  title =  { Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking},
  booktitle = {To appear in 62nd {IEEE} Conference on Decision and Control, {CDC} 2023},
  publisher = {{IEEE}},
  year      = {2023}
}
```

## Software 

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
