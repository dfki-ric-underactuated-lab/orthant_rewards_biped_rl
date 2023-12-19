---
title: Deriving Rewards for Reinforcement Learning from Symbolic Behaviour Descriptions of Bipedal Walking
github: https://github.com/dfki-ric-underactuated-lab/orthant_rewards_biped_rl/
pdf: http://arxiv.org/abs/2312.10328
authors:
  - {name: Daniel Harnack,  affiliation_key: 1}
  - {name: Christoph Lüth, affiliation_key: 1 2, link: http://www.informatik.uni-bremen.de/~clueth/}
  - {name: Lukas Gross,   affiliation_key: 1, link: https://robotik.dfki-bremen.de/de/ueber-uns/mitarbeiter/lugr02.html}
  - {name: Shivesh Kumar, affiliation_key: 1, link: https://robotik.dfki-bremen.de/de/ueber-uns/mitarbeiter/shku02.html}
  - {name: Frank Kirchner, affiliation_key: 1 2, link: https://robotik.dfki-bremen.de/de/ueber-uns/mitarbeiter/frki01.html}
affiliations:
  - {name: German Research Center for Artificial Intelligence, affiliation_key: 1, link: https://www.dfki.de/ }
  - {name: University of Bremen, affiliation_key: 2, link: https://www.uni-bremen.de/}
---

## Abstract

Generating physical movement behaviours from their symbolic description is a
long-standing challenge in artificial intelligence (AI) and robotics, 
requiring insights into numerical optimization methods as well as into
formalizations from symbolic AI and reasoning. In this paper, a novel approach
to finding a reward function from a symbolic description is proposed. The
intended system behaviour is modelled as a hybrid automaton, which reduces the
system state space to allow more efficient reinforcement learning. The
approach is applied to bipedal walking, by modelling the walking robot as a
hybrid automaton over state space orthants, and used with the compass walker
to derive a reward that incentivizes following the hybrid automaton cycle. As
a result, training times of reinforcement learning controllers are reduced
while final walking speed is increased. The approach can serve as a blueprint
how to generate reward functions from symbolic AI and reasoning.

## Presentation

![Presentation at the 62nd Conference on Decision and Control (CDC 2023), Singapore, 13.12.2023](static/slides.pdf){width=720 height=405}

## Video

<div>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/CkvLvz_tLtc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Code

The source code for the work described in the paper can be found [here](https://github.com/dfki-ric-underactuated-lab/orthant_rewards_biped_rl/).

## Citation
```
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

