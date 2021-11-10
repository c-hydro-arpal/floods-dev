# -------------------------------------------------------------------------------------
# Libraries
import logging
import os

import h5py
import numpy as np
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read hazard file in mat format
def read_file_hazard(file_name, file_vars=None):

    if file_vars is None:
        file_vars = ['mappa_h']

    if os.path.exists(file_name):

        file_collection = {}
        with h5py.File(file_name, 'r') as file_handle:
            for var_name in file_vars:
                file_data = file_handle[var_name].value
                file_data_t = np.transpose(file_data)
                file_collection[var_name] = file_data_t
    else:
        file_collection = None

    return file_collection

# -------------------------------------------------------------------------------------
