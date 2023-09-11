# get maximal rewards to be expected for the different reward setups
from path_handling.parameter_loading import load_parameters
from path_handling import project_root_dir
from plant.compass_walker import CompassWalker
from simulator.simulator import CompassWalkerSimulator
from controller.controller import VirtualGravityController
from rl_environments.compass_walker_env import CompassWalkerEnv
from pprint import pprint

init = [0.0, 0.0, -0.4, 2.0]   # theta_1, theta_2, theta_1_dot, theta_2_dot [0.0, np.pi/8.0, 0.0, 0.0]
stance_foot_coordinates = [0.0, 0.0]        # actually only the x coordinate is interesting, y should always be zero
virtual_slope = -0.07   # rad

model_parameters, sim_parameters, _ = load_parameters(project_root_dir)

# setup plant, simulator, controller
plant = CompassWalker(model_parameters, use_precalculated_dynamics=True)
simulator = CompassWalkerSimulator(plant=plant, dt=sim_parameters['dt'])
simulator.set_initial_values(stance_foot_coordinates=stance_foot_coordinates,
                             theta_1_init=init[0],
                             theta_2_init=init[1],
                             theta_1_dot_init=init[2],
                             theta_2_dot_init=init[3])

# controller = ZeroController()
controller = VirtualGravityController(plant=plant, virtual_slope=virtual_slope)

# run the simulation
_, _, control_commands_rec = simulator.simulate(controller=controller,
                                                time_span=[sim_parameters['t_start'],
                                                           sim_parameters['t_end']],
                                                record_trajectories=True)


# get the various rewards for these trajectories
rew_dict_sparse = {'action_jerkiness': -0.001,
                   'fall': -10.0,
                   'final_distance_reached': 1.0,
                   'forward_velocity': 0.00,
                   'follow_orthant_sequence': 0.00}
rew_dict_forward = {'action_jerkiness': -0.001,
                    'fall': -10.0,
                    'final_distance_reached': 1.0,
                    'forward_velocity': 0.01,
                    'follow_orthant_sequence': 0.00}
rew_dict_orthant = {'action_jerkiness': -0.001,
                    'fall': -10.0,
                    'final_distance_reached': 1.0,
                    'forward_velocity': 0.00,
                    'follow_orthant_sequence': 0.01}
rew_dict_combo = {'action_jerkiness': -0.001,
                  'fall': -10.0,
                  'final_distance_reached': 1.0,
                  'forward_velocity': 0.005,
                  'follow_orthant_sequence': 0.005}

for rew_dict in [rew_dict_sparse, rew_dict_forward, rew_dict_orthant, rew_dict_combo]:
    simulator = CompassWalkerSimulator(plant=plant, dt=sim_parameters['dt'])
    simulator.set_initial_values(stance_foot_coordinates=stance_foot_coordinates,
                                 theta_1_init=init[0],
                                 theta_2_init=init[1],
                                 theta_1_dot_init=init[2],
                                 theta_2_dot_init=init[3])

    env = CompassWalkerEnv(simulator, None, None, rew_dict, 1000, action_lim_min=None, action_lim_max=None)
    rew = 0
    for k in range(control_commands_rec.shape[1]):
        _, reward, _, _ = env.step(control_commands_rec[:, k])
        rew += reward

    rew_dict.update({'reward_gotten': rew})

pprint(rew_dict_sparse)
pprint(rew_dict_forward)
pprint(rew_dict_orthant)
pprint(rew_dict_combo)
