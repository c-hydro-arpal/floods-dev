"""
Library Features:

Name:          lib_utils_hazard
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220208'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import os

import h5py
import numpy as np

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to read hazard file in mat format
def read_file_hazard(file_name, file_vars=None, file_format=None, file_scale_factor=None):

    if file_vars is None:
        file_vars = ['mappa_h']
    if file_format is None:
        file_format = [np.float32]
    if file_scale_factor is None:
        file_scale_factor = [1]

    if os.path.exists(file_name):

        file_collection = {}
        with h5py.File(file_name, 'r') as file_handle:
            for var_name, var_format, var_scale_factor in zip(file_vars, file_format, file_scale_factor):

                file_data = file_handle[var_name][()]
                # file_data = file_handle[var_name].value  ## old syntax

                assert file_data.dtype == var_format, "Assertion failed in expected variable " + var_name + " format "

                file_data_t = np.transpose(file_data)
                file_collection[var_name] = file_data_t / var_scale_factor
    else:
        file_collection = None

    return file_collection

# -------------------------------------------------------------------------------------
