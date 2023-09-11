# custom gym environment to train compass walking
from orthants.find_orthant import find_orthant
import inspect
import gym
from gym import spaces
import numpy as np


class CompassWalkerEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator, init_state,
                 init_state_rand_range, rewards_and_weights_dict: dict, max_steps_per_episode,
                 action_lim_min=None, action_lim_max=None):
        super(CompassWalkerEnv, self).__init__()
        self.simulator = simulator
        self.init_state = init_state
        self.init_state_rand_range = init_state_rand_range
        self.max_steps_per_episode = max_steps_per_episode
        self.steps_taken_this_episode = 0
        self.cumulative_reward = 0
        self.last_hip_position = None
        self.last_action = None
        self.last_orthant = None
        self.reward_function_list = self._get_reward_functions(rewards_and_weights_dict)

        if action_lim_min is None and action_lim_max is None:
            self.action_space = spaces.Box(low=np.array([-2.0, -2.0]),
                                           high=np.array([0.0, 0.0]))
        else:
            self.action_space = spaces.Box(low=np.array(action_lim_min),
                                           high=np.array(action_lim_max))

        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -3.0, -3.0]),
                                            high=np.array([1.0, 1.0, 3.0, 3.0]))

    def step(self, action):
        self.simulator.step(action)
        self.steps_taken_this_episode += 1
        observation = self._get_observation()
        done = False
        if self.simulator.terminated or self.steps_taken_this_episode >= self.max_steps_per_episode:
            done = True

        reward = self._calculate_reward(observation, action, done, self.simulator.terminated)
        self.cumulative_reward += reward
        hip_pos = self.simulator.plant.cart_coord_hip(self.simulator.current_state,
                                                      self.simulator.current_stance_foot_coordinates)
        self.last_hip_position = hip_pos

        info = {}

        self.last_action = action

        return observation, reward, done, info

    def _get_observation(self, current_state=None, stance_foot_coordinates=None):
        if current_state is None and stance_foot_coordinates is None:
            configuration_obs = self.simulator.current_state
            com_obs = self.simulator.plant.cart_coord_hip(configuration_obs,
                                                          self.simulator.current_stance_foot_coordinates)
        else:
            configuration_obs = current_state
            com_obs = self.simulator.plant.cart_coord_hip(configuration_obs,
                                                          stance_foot_coordinates)

        observation = configuration_obs

        return observation

    def reset(self):
        self.simulator.terminated = False
        self.cumulative_reward = 0
        self.last_hip_position = None
        self.steps_taken_this_episode = 0
        self.simulator.set_initial_values(stance_foot_coordinates=[0, 0],
                                          theta_1_init=self.init_state[0],
                                          theta_2_init=self.init_state[1],
                                          theta_1_dot_init=self.init_state[2],
                                          theta_2_dot_init=self.init_state[3])
        non_noisy_init = np.array([self.simulator.theta_1_init,
                                   self.simulator.theta_2_init,
                                   self.simulator.theta_1_dot_init,
                                   self.simulator.theta_2_dot_init])
        init_with_noise = non_noisy_init
        init_with_noise += (((np.random.rand(len(non_noisy_init)) * 2) - 1) * np.array(self.init_state_rand_range))
        self.simulator.set_initial_values([0, 0],
                                          *init_with_noise)
        observation = self._get_observation()
        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _calculate_reward(self, observations, action, done, fell):
        rew = 0
        for reward_function in self.reward_function_list:
            rew += reward_function(observations, action, done, fell)
        return rew

    def _get_reward_functions(self, rewards_and_weight_dict):
        # a function to collect the known reward functions
        reward_functions_list = []
        all_class_methods = inspect.getmembers(self, predicate=inspect.ismethod)
        known_reward_functions = [cm for cm in all_class_methods if cm[0].startswith('_reward')]
        known_reward_names = [krf[0] for krf in known_reward_functions]
        for reward_name, reward_weight in rewards_and_weight_dict.items():
            reward_function_idx = known_reward_names.index(f'_reward_{reward_name}')
            weighted_reward_function = lambda obs, act, done, fell, reward_weight=reward_weight, reward_function_idx=reward_function_idx: reward_weight * known_reward_functions[reward_function_idx][1](obs, act, done, fell)
            reward_functions_list.append(weighted_reward_function)

        return reward_functions_list

    # here the reward functions
    def _reward_final_distance_reached(self, observation, action, done, fell):
        # Todo: using final stance foot x-coordinate as measure of distance travelled... best choice?
        rew = 0
        if done:
            rew = self.simulator.current_stance_foot_coordinates[0]
        return rew

    def _reward_action_jerkiness(self, observation, action, done, fell):
        rew = 0
        # only calculate meaningful reward if at least one action has been taken
        if self.last_action is not None:
            rew = np.sum(np.square(self.last_action - action))
        return rew

    def _reward_symmetry(self, observation, action, done, fell):
        # may be implemented
        return 0

    def _reward_fall(self, observation, action, done, fell):
        rew = 0
        if fell:
            rew = 1
        return rew

    def _reward_forward_velocity(self, observation, action, done, fell):
        rew = 0
        if self.last_hip_position is not None:
            hip_pos = self.simulator.plant.cart_coord_hip(self.simulator.current_state,
                                                          self.simulator.current_stance_foot_coordinates)
            if (hip_pos - self.last_hip_position)[0] > 0:
                rew = 1
            else:
                rew = -1

        return rew

    def _reward_follow_orthant_sequence(self, observation, action, done, fell):
        rew = 0
        th_1, th_2, th_1_dot, th_2_dot = observation
        if self.last_orthant is None:
            self.last_orthant = find_orthant(th_1, th_1_dot, th_2, th_2_dot)
        else:
            current_orthant = find_orthant(th_1, th_1_dot, th_2, th_2_dot)
            if current_orthant == self.last_orthant or current_orthant == np.mod(self.last_orthant + 1, 5):
                rew = 1
            else:
                rew = -1

            self.last_orthant = current_orthant

        return rew

