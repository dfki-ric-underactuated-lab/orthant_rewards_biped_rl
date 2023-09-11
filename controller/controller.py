# controllers for the compass walker
import numpy as np
import os
from stable_baselines3 import SAC, PPO
from rl_environments.compass_walker_env import CompassWalkerEnv
from simulator.simulator import CompassWalkerSimulator
from plant.compass_walker import CompassWalker
from path_handling.parameter_loading import load_parameters


# a controller doing nothing
class ZeroController:
    def __init__(self):
        self.control_dim = 2

    def get_control_input(self, state, stance_foot_coordinates):
        u = np.zeros(self.control_dim)
        return u


# The following are some controllers described in
# Asano et al. (2005) Biped Gait Generation and Control Based on a Unified Property of
# Passive Dynamic Walking, https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1492492

# The first is a controller that emulates gravity of a slope, i.e. is passive
class VirtualGravityController:
    def __init__(self, plant: CompassWalker, virtual_slope=0.0265):
        self.phi = virtual_slope  # rad
        self.m_h = plant.mass_hip
        self.m = plant.mass_legs
        self.l = plant.length_legs
        self.a = plant.length_a
        self.b = plant.length_b
        self.g = plant.earth_gravity

    def get_control_input(self, state, stance_foot_coordinates):
        theta_1 = state[0]
        theta_2 = state[2]
        u_1 = (self.m_h * self.l + self.m * (self.a + self.l))*np.cos(theta_1) - self.m * self.b * np.cos(theta_2)
        u_2 = self.m * self.b * np.cos(theta_2)

        u_vec = np.array([u_1, u_2]) * self.g * np.tan(self.phi)

        return u_vec


class RLController:
    def __init__(self, model_data_folder=None, ep_to_load='final'):
        self.model, self.env = self._load_model(model_data_folder, ep_to_load)

    def _load_model(self, model_data_folder, ep_to_load):
        if model_data_folder is None:
            raise FileNotFoundError(f'The model data folder {model_data_folder} does not exist')
        else:
            model_parameters, sim_parameters, rl_parameters = load_parameters(model_data_folder)

            # The env is needed to load the model ... a bit stupid
            plant = CompassWalker(model_parameters, use_precalculated_dynamics=True)
            simulator = CompassWalkerSimulator(plant=plant, dt=sim_parameters['dt'])

            env = CompassWalkerEnv(simulator, init_state=rl_parameters['init_state'],
                                   init_state_rand_range=rl_parameters['init_state_rand_range'],
                                   rewards_and_weights_dict=rl_parameters['reward_setup'],
                                   max_steps_per_episode=int(sim_parameters['t_end'] / sim_parameters['dt']),
                                   action_lim_min=rl_parameters['action_lim_min'],
                                   action_lim_max=rl_parameters['action_lim_max'])

        model_type = rl_parameters['algorithm']

        if model_type == 'SAC':
            model_class = SAC
        elif model_type == 'PPO':
            model_class = PPO
        if ep_to_load == 'final':
            model = model_class.load(os.path.join(model_data_folder, 'model_final.zip'), device='cpu')
        elif ep_to_load == 'best':
            model = model_class.load(os.path.join(model_data_folder, 'best_model.zip'), device='cpu')
        else:
            model = model_class.load(os.path.join(model_data_folder, f'model_{ep_to_load}_steps.zip'), device='cpu')

        model.policy.set_training_mode(False)

        return model, env

    def get_control_input(self, state, stance_foot_coordinates):
        observation = self.env._get_observation(state, stance_foot_coordinates)
        u_vec, _ = self.model.policy.predict(observation, deterministic=True)

        return u_vec

