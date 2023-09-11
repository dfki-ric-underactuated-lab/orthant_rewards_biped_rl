# a function to classify orthants

import numpy as np


def find_orthant(th_1, th_1_dot, th_2, th_2_dot):
    walking_orthant_num = -1  # -1 is the default orthant, meaning not part of the walking cycle

    walking_orthant_1_signs = np.array([1, -1, -1, 1])
    walking_orthant_2_signs = np.array([1, -1, 1, 1])
    walking_orthant_3_signs = np.array([-1, -1, 1, 1])
    walking_orthant_4_signs = np.array([-1, -1, 1, -1])

    state_orthant = np.sign(np.array([th_1, th_1_dot, th_2, th_2_dot]))

    if sum(state_orthant == walking_orthant_1_signs) == len(state_orthant):
        walking_orthant_num = 1
    elif sum(state_orthant == walking_orthant_2_signs) == len(state_orthant):
        walking_orthant_num = 2
    elif sum(state_orthant == walking_orthant_3_signs) == len(state_orthant):
        walking_orthant_num = 3
    elif sum(state_orthant == walking_orthant_4_signs) == len(state_orthant):
        walking_orthant_num = 4

    return walking_orthant_num

