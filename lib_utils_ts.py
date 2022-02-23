"""
Library Features:

Name:          lib_utils_ts
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220223'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
from copy import deepcopy

from lib_utils_io import save_file_json
from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to prepare time-series in json format
def prepare_file_ts(file_keys, file_data, file_info):
    collections_ts = {}
    for key_step in file_keys:
        if key_step in list(file_data.keys()):
            dframe_data = file_data[key_step]
            obj_data = dframe_data.to_dict(orient='list')
        else:
            obj_data = None
            log_stream.warning(' ===> Variable "file_data" is not available for section "' + key_step + '"')
        if key_step in list(file_info.keys()):
            obj_info = file_info[key_step]
        else:
            obj_info = None
            log_stream.warning(' ===> Variable "file_info" is not available for section "' + key_step + '"')

        obj_ts = None
        if (obj_info is not None) and (obj_data is not None):
            obj_ts = {**obj_info, **obj_data}
        elif (obj_info is None) and (obj_data is not None):
            obj_ts = deepcopy(obj_data)
        elif (obj_info is not None) and (obj_data is None):
            obj_ts = deepcopy(obj_info)
        elif (obj_info is None) and (obj_data is None):
            obj_ts = None

        collections_ts[key_step] = obj_ts

    return collections_ts

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to save time-series in json format
def save_file_ts(file_name, file_data_collection):
    save_file_json(file_name, file_data_dict=file_data_collection)
# -------------------------------------------------------------------------------------
