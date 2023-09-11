# a function for training on the compass walker environment

import json
from rl_environments.compass_walker_env import CompassWalkerEnv
from plant.compass_walker import CompassWalker
from simulator.simulator import CompassWalkerSimulator
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
from path_handling import project_root_dir
from path_handling.parameter_loading import load_parameters
from path_handling.make_log_folder import make_log_folder


def compass_walker_training(n_episodes=1, n_steps_max=1000, save_steps_interval=None, max_steps_per_episode=1000):
    # first get some parameters
    # model parameters
    model_parameters, sim_parameters, rl_parameters = load_parameters(project_root_dir)

    # make a log folder
    save_folder = make_log_folder()

    plant = CompassWalker(model_parameters, use_precalculated_dynamics=True)
    simulator = CompassWalkerSimulator(plant=plant, dt=sim_parameters['dt'], verbose=False)
    env = CompassWalkerEnv(simulator, init_state=rl_parameters['init_state'],
                           init_state_rand_range=rl_parameters['init_state_rand_range'],
                           rewards_and_weights_dict=rl_parameters['reward_setup'],
                           max_steps_per_episode=max_steps_per_episode,
                           action_lim_min=rl_parameters['action_lim_min'],
                           action_lim_max=rl_parameters['action_lim_max'])

    eval_env = CompassWalkerEnv(simulator, init_state=rl_parameters['init_state'],
                                init_state_rand_range=rl_parameters['init_state_rand_range'],
                                rewards_and_weights_dict=rl_parameters['reward_setup'],
                                max_steps_per_episode=max_steps_per_episode,
                                action_lim_min=rl_parameters['action_lim_min'],
                                action_lim_max=rl_parameters['action_lim_max'])

    # add a monitor to log stuff
    env = Monitor(env, save_folder)

    if rl_parameters['algorithm'] == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cpu')
    elif rl_parameters['algorithm'] == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, device='cpu')
    else:
        raise NotImplementedError(f'Algorithm choice {rl_parameters["algorithm"]} not recognized or implemented yet :<')

    if save_steps_interval is not None:
        checkpoint_callback = CheckpointCallback(save_freq=save_steps_interval,
                                                 save_path=save_folder,
                                                 name_prefix='model')
        # Use deterministic actions for evaluation
        eval_callback = EvalCallback(eval_env, best_model_save_path=save_folder, eval_freq=save_steps_interval,
                                     deterministic=True, render=False)
        callbacks = CallbackList([checkpoint_callback, eval_callback])
        model.learn(total_timesteps=n_steps_max*n_episodes, callback=callbacks)
    else:
        model.learn(total_timesteps=n_steps_max*n_episodes)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'eval performance: mean {mean_reward}, std {std_reward}')
    model.save(os.path.join(save_folder, 'model_final'))


def compass_walker_eval(path_to_model, max_steps_per_episode=1000, ep_to_load='final'):
    model_parameters, sim_parameters, rl_parameters = load_parameters(path_to_model)

    plant = CompassWalker(model_parameters, use_precalculated_dynamics=True)
    simulator = CompassWalkerSimulator(plant=plant, dt=sim_parameters['dt'], verbose=False)
    env = CompassWalkerEnv(simulator, init_state=rl_parameters['init_state'],
                           init_state_rand_range=rl_parameters['init_state_rand_range'],
                           rewards_and_weights_dict=rl_parameters['reward_setup'],
                           max_steps_per_episode=max_steps_per_episode)

    if rl_parameters['algorithm'] == 'SAC':
        model_class = SAC
    elif rl_parameters['algorithm'] == 'PPO':
        model_class = PPO
    else:
        raise NotImplementedError(f'Algorithm choice {rl_parameters["algorithm"]} not recognized or implemented yet :<')

    if ep_to_load == 'final':
        model = model_class.load(os.path.join(path_to_model, 'model_final.zip'), device='cpu')
    else:
        model = model_class.load(os.path.join(path_to_model, f'model_{ep_to_load}_steps.zip'), device='cpu')


    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f'Evaluation: mean {mean_reward}, std {std_reward}')    # evaluation loop
    obs = env.reset()
    states_rec = np.zeros((4, max_steps_per_episode))
    stance_foot_coordinates_rec = np.zeros((2, max_steps_per_episode))
    control_commands_rec = np.zeros((2, max_steps_per_episode))

    for i in range(max_steps_per_episode):
        action, _state = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        states_rec[:, i] = env.simulator.current_state
        stance_foot_coordinates_rec[:, i] = env.simulator.current_stance_foot_coordinates
        control_commands_rec[:, i] = action

        if env.simulator.terminated:
            states_rec = states_rec[:, :i]
            stance_foot_coordinates_rec = stance_foot_coordinates_rec[:, :i]
            control_commands_rec = control_commands_rec[:, :i]
            break

    return states_rec, stance_foot_coordinates_rec, control_commands_rec


