# this is the main simulation plant for the compass walker

import numpy as np
from plant.compass_walker import CompassWalker
import scipy.integrate as integrate


class CompassWalkerSimulator:
    def __init__(self, plant: CompassWalker, dt, verbose=False):
        self.plant = plant
        self.t = 0.0
        self.dt = dt
        self.verbose = verbose
        self.terminated = False  # Flag set to true if terminal fall occurred
        # to be filled by set_initial_values()
        self.current_stance_foot_coordinates = None
        self.theta_1_init = None
        self.theta_2_init = None
        self.theta_1_dot_init = None
        self.theta_2_dot_init = None

        self.current_state = [self.theta_1_init,
                              self.theta_1_init,
                              self.theta_1_init,
                              self.theta_1_init]

    def set_initial_values(self, stance_foot_coordinates,
                           theta_1_init, theta_2_init, theta_1_dot_init, theta_2_dot_init):
        self.current_stance_foot_coordinates = stance_foot_coordinates
        self.theta_1_init = theta_1_init
        self.theta_2_init = theta_2_init
        self.theta_1_dot_init = theta_1_dot_init
        self.theta_2_dot_init = theta_2_dot_init

        self.current_state = [self.theta_1_init,
                              self.theta_2_init,
                              self.theta_1_dot_init,
                              self.theta_2_dot_init]

    # a single simulation step
    def step(self, control_input):
        dyn = self.plant.get_dynamics(control_input)

        current_time = self.t
        target_time = self.t + self.dt
        last_foot_change = None
        while current_time != target_time and not self.terminated:
            events = self.plant.get_events(self.current_stance_foot_coordinates)

            integrated = integrate.solve_ivp(dyn, (current_time, target_time),
                                             self.current_state,
                                             events=events)

            # check for events, i.e. foot hits and falls
            x_temp = integrated.y.transpose()
            if len(integrated.y_events[0]) == 0 and len(integrated.y_events[1]) == 0:
                self.current_state = x_temp[-1, :]
            else:
                event_dyn_0, event_dyn_1 = self.plant.get_event_dynamics(x_temp[-1, :],
                                                                         self.current_stance_foot_coordinates)
                if len(integrated.y_events[0]) != 0:
                    if self.verbose:
                        print('Change of foot happened!')
                    x_switched, stance_foot_coordinates_switched = event_dyn_0(x_temp[-1, :])
                    self.current_state = x_switched
                    self.current_stance_foot_coordinates = stance_foot_coordinates_switched
                    time_foot_change = integrated.t[-1]
                    if last_foot_change is not None:
                        if time_foot_change - last_foot_change < self.dt/10:
                            if self.verbose:
                                print('Wobbling in place! Abort, Abort!')
                            self.terminated = True
                            continue
                    else:
                        last_foot_change = time_foot_change
                if len(integrated.y_events[1]) != 0:
                    if self.verbose:
                        print('Fell! Abort, Abort!')
                    self.terminated = True
                    continue

            current_time = integrated.t[-1]

    # simulate for several steps
    def simulate(self, controller, time_span, record_trajectories=True, init_values=None):
        t = np.arange(time_span[0], time_span[1], self.dt)
        if record_trajectories:
            x_rec = np.zeros((len(self.current_state),
                              len(t)))
            stance_foot_coord_rec = np.zeros((len(self.current_stance_foot_coordinates),
                                              len(t)))
            control_commands_rec = np.zeros((2, len(t)))
        else:
            x_rec = None
            stance_foot_coord_rec = None
            control_commands_rec = None

        t_idx = 0
        while not self.terminated and t_idx <= len(t)-1:
            u = controller.get_control_input(self.current_state, self.current_stance_foot_coordinates)
            self.step(control_input=u)

            if record_trajectories:
                x_rec[:, t_idx] = self.current_state
                stance_foot_coord_rec[:, t_idx] = self.current_stance_foot_coordinates
                control_commands_rec[:, t_idx] = u

            t_idx += 1

        # crop the recordings
        if self.terminated:
            x_rec = x_rec[:, :t_idx]
            stance_foot_coord_rec = stance_foot_coord_rec[:, :t_idx]
            control_commands_rec = control_commands_rec[:, :t_idx]

        return x_rec, stance_foot_coord_rec, control_commands_rec

