# script to train rl agent

from rl_training.trainer import compass_walker_training

n_trials = 1
n_episodes = 1
n_steps_max = 100  # 500000
max_steps_per_episode = 1000

for k in range(n_trials):
    compass_walker_training(n_episodes=n_episodes,
                            n_steps_max=n_steps_max,
                            save_steps_interval=10000,
                            max_steps_per_episode=max_steps_per_episode)

    print(f'Training trial {k} finished.')
