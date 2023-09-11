# this is the main simulation plant for the compass walker

import sympy as sym
import numpy as np
from functools import partial


class CompassWalker:
    def __init__(self, model_parameters, use_precalculated_dynamics=True):
        # general symbols
        th_1, th_2, th_1_dot, th_2_dot, u_1, u_2, t = sym.symbols('th_1, th_2, th_1_dot, th_2_dot, u_1, u_2, t')
        m_h, m, a, b, g = sym.symbols('m_h, m, a, b, g')
        l = a + b

        # dynamics
        th = sym.Matrix(2, 1, [th_1, th_2])
        th_dot = sym.Matrix(2, 1, [th_1_dot, th_2_dot])
        u = sym.Matrix(2, 1, [u_1, u_2])
        M = sym.Matrix(2, 2, [(m_h + m) * l * l + m * a * a, -m * b * l * sym.cos(th_1 - th_2),
                              -m * b * l * sym.cos(th_1 - th_2), m * b * b])
        C = sym.Matrix(2, 2, [0, -m * b * l * sym.sin(th_1 - th_2) * th_2_dot,
                              m * b * l * sym.sin(th_1 - th_2) * th_1_dot, 0])
        G = sym.Matrix(2, 1, [-(m_h * l + m * a + m * l) * g * sym.sin(th_1), m * b * g * sym.sin(th_2)])
        S = sym.Matrix(2, 2, [1, 1,
                              0, -1])

        # leg impact transition
        Q_after = sym.Matrix(2, 2, [m_h * l * l + m * a * a + m * l * (l - b * sym.cos(th_1 - th_2)),
                                    m * b * (b - l * sym.cos(th_1 - th_2)),
                                    -m * b * l * sym.cos(th_1 - th_2), m * b * b])
        Q_before = sym.Matrix(2, 2, [(m_h * l * l + 2 * m * a * l) * sym.cos(th_1 - th_2) - m * a * b, -m * a * b,
                                     -m * a * b, 0])

        if not use_precalculated_dynamics:
            print('Calculating symbolic dynamics. This might take a while...')
            # Equation for acceleration of q:
            q_ddot = sym.simplify(M.inv() * (S * u - C * th_dot - G))
            # Equation for impulse transition
            q_dot_trans = sym.simplify(Q_after.inv() * Q_before * th_dot)
            print('done')
        else:
            print('Using precaculated dynamics...')
            q_ddot = sym.Matrix(2, 1, [(b * (b * m * th_2_dot ** 2 * (a + b) * sym.sin(th_1 - th_2) + g * (
                        a * m + m * (a + b) + m_h * (a + b)) * sym.sin(th_1) + u_1 + u_2) - (a + b) * (
                                                    b * g * m * sym.sin(th_2) + b * m * th_1_dot ** 2 * (
                                                        a + b) * sym.sin(th_1 - th_2) + u_2) * sym.cos(th_1 - th_2)) / (
                                                   b * (-a ** 2 * m * sym.cos(
                                               th_1 - th_2) ** 2 + 2 * a ** 2 * m + a ** 2 * m_h - 2 * a * b * m * sym.cos(
                                               th_1 - th_2) ** 2 + 2 * a * b * m + 2 * a * b * m_h - b ** 2 * m * sym.cos(
                                               th_1 - th_2) ** 2 + b ** 2 * m + b ** 2 * m_h)),
                                       (b * m * (a + b) * (
                                                   b * m * th_2_dot ** 2 * (a + b) * sym.sin(th_1 - th_2) + g * (
                                                       a * m + m * (a + b) + m_h * (a + b)) * sym.sin(
                                               th_1) + u_1 + u_2) * sym.cos(th_1 - th_2) - (
                                                    b * g * m * sym.sin(th_2) + b * m * th_1_dot ** 2 * (
                                                        a + b) * sym.sin(th_1 - th_2) + u_2) * (
                                                    2 * a ** 2 * m + a ** 2 * m_h + 2 * a * b * m + 2 * a * b * m_h + b ** 2 * m + b ** 2 * m_h)) / (
                                                   b ** 2 * m * (-a ** 2 * m * sym.cos(
                                               th_1 - th_2) ** 2 + 2 * a ** 2 * m + a ** 2 * m_h - 2 * a * b * m * sym.cos(
                                               th_1 - th_2) ** 2 + 2 * a * b * m + 2 * a * b * m_h - b ** 2 * m * sym.cos(
                                               th_1 - th_2) ** 2 + b ** 2 * m + b ** 2 * m_h))])
            self.q_ddot = q_ddot

            q_dot_trans = sym.Matrix(2, 1, [(-a * b * m * th_2_dot + th_1_dot * (
                        a ** 2 * m + a ** 2 * m_h + a * b * m + 2 * a * b * m_h + b ** 2 * m_h) * sym.cos(
                th_1 - th_2)) / (-a ** 2 * m * sym.cos(
                th_1 - th_2) ** 2 + 2 * a ** 2 * m + a ** 2 * m_h - 2 * a * b * m * sym.cos(
                th_1 - th_2) ** 2 + 2 * a * b * m + 2 * a * b * m_h - b ** 2 * m * sym.cos(
                th_1 - th_2) ** 2 + b ** 2 * m + b ** 2 * m_h),
                                            (-a * b * m * th_2_dot * (a + b) * sym.cos(th_1 - th_2) - th_1_dot * (a * (
                                                        2 * a ** 2 * m + a ** 2 * m_h - a * b * m * sym.cos(
                                                    th_1 - th_2) + 2 * a * b * m + 2 * a * b * m_h - b ** 2 * m * sym.cos(
                                                    th_1 - th_2) + b ** 2 * m + b ** 2 * m_h) + (a + b) * (a * b * m - (
                                                        a + b) * (2 * a * m + m_h * (a + b)) * sym.cos(
                                                th_1 - th_2)) * sym.cos(th_1 - th_2))) / (b * (-a ** 2 * m * sym.cos(
                                                th_1 - th_2) ** 2 + 2 * a ** 2 * m + a ** 2 * m_h - 2 * a * b * m * sym.cos(
                                                th_1 - th_2) ** 2 + 2 * a * b * m + 2 * a * b * m_h - b ** 2 * m * sym.cos(
                                                th_1 - th_2) ** 2 + b ** 2 * m + b ** 2 * m_h))])

        self.mass_hip = model_parameters['mass_hip']        # kg
        self.mass_legs = model_parameters['mass_legs']      # kg
        self.length_a = model_parameters['length_a']        # m
        self.length_b = model_parameters['length_b']        # m
        self.length_legs = self.length_a + self.length_b    # m
        self.earth_gravity = model_parameters['gravity']    # m/s^2

        self.conf = {m_h: self.mass_hip,
                     m: self.mass_legs,
                     a: self.length_a,
                     b: self.length_b,
                     g: self.earth_gravity
                     }

        print('Lambdifying symbolic dynamics... ')
        s_swing_phase = (th_1, th_2, th_1_dot, th_2_dot, u_1, u_2)
        self.sec_der = sym.lambdify(s_swing_phase, q_ddot.subs(self.conf), modules='numpy')
        s_transition = (th_1, th_2, th_1_dot, th_2_dot)
        self.trans = sym.lambdify(s_transition, q_dot_trans.subs(self.conf), modules='numpy')

    # some helper functions
    def q(self, x):
        return np.array(x[:2])

    def q_dot(self, x):
        return np.array(x[2:])

    def dyn(self, t, x, u):
        out = np.zeros(x.shape)
        out[:2] = self.q_dot(x)
        # out[2:] = sec_der(*x, u(x))[:, 0]
        out[2:] = self.sec_der(*x, *u)[:, 0]
        return out

    def leg_transition(self, x, tr):
        tr_plus = self.cart_coord_foot(x, tr)
        out = np.zeros(x.shape)
        out[:2] = np.flip(self.q(x))
        out[2:] = self.trans(x[0], x[1], x[2], x[3])[:, 0]
        return out, tr_plus

    # transformations to cartesian coordinates, tr gives stance foot coordinates as (x, y)
    def cart_coord_hip(self, x, tr):
        return tr + self.length_legs * np.array([-np.sin(x[0]), np.cos(x[0])])

    def cart_coord_stance_leg(self, x, tr):
        return tr + self.length_a * np.array([-np.sin(x[0]), np.cos(x[0])])

    def cart_coord_swing_leg(self, x, tr):
        return self.cart_coord_hip(x, tr) + self.length_b * np.array([np.sin(x[1]), -np.cos(x[1])])

    def cart_coord_foot(self, x, tr):
        return self.cart_coord_hip(x, tr) + self.length_legs * np.array([np.sin(x[1]), -np.cos(x[1])])

    def cart_coord_hip_der(self, x, tr):
        return self.length_legs * np.array([-np.cos(x[0]), -np.sin(x[0])]) * x[2]

    def cart_coord_foot_der(self, x, tr):
        return self.cart_coord_hip_der(x, tr) + self.length_legs * np.array([np.cos(x[1]), np.sin(x[1])]) * x[3]

    # collision detection
    def hip_height(self, t, x, tr):
        return self.cart_coord_hip(x, tr)[1]  # FixMe: here the floor could be added

    def swing_leg_height(self, t, x, tr):
        return self.cart_coord_swing_leg(x, tr)[1]

    def minimal_height(self, t, x, tr):
        return min(self.hip_height(t, x, tr), self.swing_leg_height(t, x, tr))

    def swing_before_stance(self, x):
        return x[0] < x[1]

    def foot_height(self, t, x, tr):
        if self.swing_before_stance(x):
            out = self.cart_coord_foot(x, tr)[1]
        else:
            out = -1
        return out

    def get_dynamics(self, u):
        def dynamics_explicit(t, x, u):
            out = np.zeros(x.shape)
            out[:2] = self.q_dot(x)
            # out[2:] = sec_der(*x, u(x))[:, 0]
            out[2:] = self.sec_der(*x, *u)[:, 0]
            return out
        dynamics = partial(dynamics_explicit, u=u)
        return dynamics

    def get_events(self, tr):
        hit = partial(self.foot_height, tr=tr)
        hit.terminal = True
        hit.direction = -1
        fall = partial(self.hip_height, tr=tr)
        fall.terminal = True

        return [hit, fall]

    def terminate(self):
        return True

    def get_event_dynamics(self, x, tr):
        leg_change = partial(self.leg_transition, tr=tr)

        return leg_change, self.terminate

