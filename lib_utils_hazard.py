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
from scipy.io import loadmat

from lib_utils_io import read_file_tif
from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
# Debug
import matplotlib.pylab as plt
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to read hazard file in mat or tiff format
def read_file_hazard(file_name, file_vars=None, file_format=None, file_scale_factor=None):

    if file_vars is None:
        file_vars = ['mappa_h']
    if file_format is None:
        file_format = [np.float32]
    if file_scale_factor is None:
        file_scale_factor = [1]

    if os.path.exists(file_name):

        if file_name.endswith('mat'):

            file_collection = {}
            try:
                with h5py.File(file_name, 'r') as file_handle:
                    for var_name, var_format, var_scale_factor in zip(file_vars, file_format, file_scale_factor):

                        if var_name in list(file_handle.keys()):
                            file_data = file_handle[var_name][()]
                            # file_data = file_handle[var_name].value  ## old syntax

                            assert file_data.dtype == var_format, "Assertion failed in expected variable " + \
                                                                  var_name + " format "

                            file_data_t = np.transpose(file_data)
                            file_collection[var_name] = file_data_t / var_scale_factor
                        else:
                            log_stream.error(' ===> Variable "' + var_name + '" not found in "' + file_name + '"')
                            raise ValueError('Variable not found')

            except BaseException as b_exp_mat:

                log_stream.warning(' ===> Read "' + file_name +
                                   '" with hdf5 library raises an exception. Try using scipy')
                log_stream.warning(' ===> Exception "' + str(b_exp_mat) + '"')

                file_handle = loadmat(file_name)

                for var_name, var_format, var_scale_factor in zip(file_vars, file_format, file_scale_factor):
                    if var_name in list(file_handle.keys()):
                        file_data = file_handle[var_name]

                        assert file_data.dtype == var_format, "Assertion failed in expected variable " + var_name + " format "
                        file_collection[var_name] = file_data / var_scale_factor
                    else:
                        log_stream.error(' ===> Variable "' + var_name + '" not found in "' + file_name + '"')
                        raise ValueError('Variable not found')

            except BaseException as b_exp_scipy:
                log_stream.warning(' ===> Read "' + file_name + '" with scipy library raises an exception')
                log_stream.warning(' ===> Exception "' + str(b_exp_scipy) + '"')
                log_stream.error(' ===> Open "' + file_name + '" failed.')
                raise IOError('File format is not readable from both hdf5 and scipy libraries')

        elif file_name.endswith('tif') or file_name.endswith('tiff'):

            file_data, file_proj, file_geotrans = read_file_tif(file_name)

            if file_data.ndim == 2:
                if file_vars.__len__() == 1:
                    var_name = file_vars[0]
                else:
                    log_stream.error(
                        ' ===> Hazard data are in 2D format; expected "var_name" length must be equal to 1')
                    raise NotImplementedError('Case not implemented yet')
                if file_format.__len__() == 1:
                    var_format = file_format[0]
                else:
                    log_stream.error(
                        ' ===> Hazard data are in 2D format; expected "var_format" length must be equal to 1')
                    raise NotImplementedError('Case not implemented yet')
                if file_scale_factor.__len__() == 1:
                    var_scale_factor = file_scale_factor[0]
                else:
                    log_stream.error(
                        ' ===> Hazard data are in 2D format; expected "var_scale_factor" length must be equal to 1')
                    raise NotImplementedError('Case not implemented yet')
            else:
                log_stream.error(' ===> Hazard data must be in 2D format')
                raise NotImplementedError('Case not implemented yet')

            assert file_data.dtype == var_format, "Assertion failed in expected variable " + var_name + " format "

            file_collection = {var_name: file_data / var_scale_factor}

        else:
            log_stream.error(' ===> Hazard file must be in "mat" or "tif/tiff" format')
            raise NotImplementedError('Case not implemented yet')

    else:
        log_stream.warning(' ===> Open "' + file_name + '" failed. File not found')
        file_collection = None

    return file_collection

# -------------------------------------------------------------------------------------
