# run and inspect agents that have been trained

import os
from path_handling import project_root_dir
import numpy as np
import matplotlib.pyplot as plt
from path_handling.parameter_loading import load_parameters
from plant.compass_walker import CompassWalker
from simulator.simulator import CompassWalkerSimulator
from controller.controller import RLController
from visualization.animation import AnimatorCompassWalker

# insert the path for your trained agent here
model_path = os.path.join(project_root_dir, 'results', 'trained_agents', '20XX_XXXXX')
ep_to_load = 'final'

# model parameters
model_parameters, sim_parameters, rl_parameters = load_parameters(model_path)

# setup plant, simulator, controller
init = rl_parameters['init_state']
stance_foot_coordinates = [0.0, 0.0]  # Those are basically always the same without loss of generality
plant = CompassWalker(model_parameters, use_precalculated_dynamics=True)
simulator = CompassWalkerSimulator(plant=plant, dt=sim_parameters['dt'])
simulator.set_initial_values(stance_foot_coordinates=stance_foot_coordinates,
                             theta_1_init=init[0],
                             theta_2_init=init[1],
                             theta_1_dot_init=init[2],
                             theta_2_dot_init=init[3])

controller = RLController(model_data_folder=model_path,
                          ep_to_load=ep_to_load)

# run the simulation
states_rec, stance_foot_coordinates_rec, control_commands_rec = simulator.simulate(controller=controller,
                                                                                   time_span=[sim_parameters['t_start'],
                                                                                              sim_parameters['t_end']],
                                                                                   record_trajectories=True)

step_idcs = np.where(np.diff(stance_foot_coordinates_rec[0, :]))[0]
foot_height = []
for x, tr in zip(states_rec.T, stance_foot_coordinates_rec.T):
    foot_height.append(plant.foot_height(None, x, tr))

plt.figure()
plt.plot(np.arange(len(states_rec[0, :])) * sim_parameters['dt'], states_rec[0, :], label='theta 1')
plt.plot(np.arange(len(states_rec[0, :])) * sim_parameters['dt'], states_rec[1, :], label='theta 2')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('angle [rad]')

plt.figure()
plt.plot(np.arange(len(control_commands_rec[0, :])) * sim_parameters['dt'], control_commands_rec[0, :], label='u 1')
plt.plot(np.arange(len(control_commands_rec[0, :])) * sim_parameters['dt'], control_commands_rec[1, :], label='u 2')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('torque [Nm]')
plt.show()


# find stance foot coordinate changes
id_1 = 0
id_2 = 1
start_t_idx = 0
stance_leg_theta = []
stance_leg_theta_dot = []
for step_idx in step_idcs:
    stance_leg_theta += (list(states_rec[id_1, start_t_idx:step_idx]))
    stance_leg_theta_dot += (list(states_rec[id_1+2, start_t_idx:step_idx]))
    id_1 = np.mod(id_1 + 1, 2)
    start_t_idx = step_idx

plt.figure()
plt.plot(stance_leg_theta, stance_leg_theta_dot)
plt.show()

# run visualization
animation = AnimatorCompassWalker(plant=plant,
                                  dt=sim_parameters['dt'],
                                  stance_foot_pos_rec=stance_foot_coordinates_rec,
                                  x_rec=states_rec)
ani = animation.animate()
plt.show()

