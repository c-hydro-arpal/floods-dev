"""
Library Features:

Name:          lib_utils_tr
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220208'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import numpy as np
from numpy.linalg import lstsq

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to compute tr exp mode
def cmp_tr_exp(section_discharge_idx, section_discharge_value):

    section_scenario_tr = np.round(np.exp(
        (section_discharge_idx * 0.5239 + section_discharge_value) / (section_discharge_idx * 1.0433)))

    section_scenario_tr_rounded, section_scenario_tr_right, section_scenario_tr_left, \
        section_scenario_weight_right, section_scenario_weight_left = cmp_tr_weights(section_scenario_tr)

    return section_scenario_tr_rounded, section_scenario_tr_right, section_scenario_tr_left, \
        section_scenario_weight_right, section_scenario_weight_left

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to compute tr linear mode
def cmp_tr_linear(section_discharge_idx, section_discharge_value, section_discharge_factor=1.16):

    section_scenario_tr_min = 0.0
    section_scenario_tr_max = np.exp(
        (section_discharge_idx * 0.5239 + section_discharge_idx * section_discharge_factor) / (section_discharge_idx * 1.0433))

    section_discharge_idx_min = 0.0
    section_discharge_idx_max = section_discharge_idx * section_discharge_factor

    points_coords = [(section_discharge_idx_min, section_scenario_tr_min),
                     (section_discharge_idx_max, section_scenario_tr_max)]
    points_x, points_y = zip(*points_coords)
    line_a = np.vstack([points_x, np.ones(len(points_x))]).T

    line_m, line_c = lstsq(line_a, points_y)[0]

    section_scenario_tr = line_m * section_discharge_value + line_c

    section_scenario_tr_rounded, section_scenario_tr_right, section_scenario_tr_left, \
        section_scenario_weight_right, section_scenario_weight_left = cmp_tr_weights(section_scenario_tr)

    return section_scenario_tr_rounded, \
        section_scenario_tr_right, section_scenario_tr_left, \
        section_scenario_weight_right, section_scenario_weight_left
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to compute tr values and weights
def cmp_tr_weights(value_tr, decimal_tr=1):

    section_scenario_tr_rounded = np.around(value_tr, decimals=decimal_tr)
    section_scenario_tr_left = np.ceil(section_scenario_tr_rounded)
    section_scenario_tr_right = np.floor(section_scenario_tr_rounded)

    section_scenario_weight_right = np.around(1 - (section_scenario_tr_rounded - section_scenario_tr_right), decimals=decimal_tr)
    section_scenario_weight_left = np.around(1 - (section_scenario_tr_left - section_scenario_tr_rounded), decimals=decimal_tr)

    return section_scenario_tr_rounded, \
        section_scenario_tr_right, section_scenario_tr_left, section_scenario_weight_right, section_scenario_weight_left
# -------------------------------------------------------------------------------------
