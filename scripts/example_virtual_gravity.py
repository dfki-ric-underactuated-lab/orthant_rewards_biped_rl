# run virtual gravity comparison controller

import numpy as np
import matplotlib.pyplot as plt
from path_handling.parameter_loading import load_parameters
from path_handling import project_root_dir
from plant.compass_walker import CompassWalker
from simulator.simulator import CompassWalkerSimulator
from controller.controller import VirtualGravityController
from visualization.animation import AnimatorCompassWalker
from matplotlib.animation import FFMpegWriter

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
plt.plot(np.arange(len(states_rec[0, :])) * sim_parameters['dt'], np.zeros((len(states_rec[0, :]), 1)))
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('angle [rad]')
plt.show()
plt.savefig(f'../results/virtual_gravity_controller/virtual_slope_{virtual_slope}_init_{init}_angles.pdf')

plt.figure()
plt.plot(np.arange(len(control_commands_rec[0, :])) * sim_parameters['dt'], control_commands_rec[0, :], label='u 1')
plt.plot(np.arange(len(control_commands_rec[0, :])) * sim_parameters['dt'], control_commands_rec[1, :], label='u 2')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('torque [Nm]')
plt.show()

# run visualization
animation = AnimatorCompassWalker(plant=plant,
                                  dt=sim_parameters['dt'],
                                  stance_foot_pos_rec=stance_foot_coordinates_rec,
                                  x_rec=states_rec)
ani = animation.animate()
plt.show()
ani.save(f'../results/virtual_gravity_controller/virtual_slope_{virtual_slope}_init_{init}.mp4',
         writer=FFMpegWriter(fps=100))

