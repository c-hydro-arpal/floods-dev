"""
Library Features:

Name:          lib_utils_hydro
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220208'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import os
import re

from copy import deepcopy
from scipy.io import loadmat
from datetime import datetime

import numpy as np
import pandas as pd

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to read info file in ascii format
def read_file_info(file_name, file_id, file_header=None, file_skip_rows=1,
                    index_name=0, index_group_start=1, index_group_end=10):

    df_data_raw = pd.read_table(file_name, header=file_header, skiprows=file_skip_rows)

    name_list = list(df_data_raw.iloc[:, index_name])
    id_list = [file_id] * name_list.__len__()

    data_tmp = df_data_raw.iloc[:, index_group_start:index_group_end].values

    data_collections = []
    for data_step in data_tmp:

        data_parsed = []
        for data_tmp in data_step:
            if isinstance(data_tmp, str):
                data_str = data_tmp.split(' ')
                for data_char in data_str:
                    data_parsed.append(int(data_char))
            else:
                data_parsed.append(data_tmp)

        data_collections.append(data_parsed)

    dict_data = {}
    for name_step, id_step, data_step in zip(name_list, id_list, data_collections):
        dict_data[name_step] = {}
        dict_data[name_step]['id'] = id_step
        dict_data[name_step]['dataset'] = data_step

    return dict_data
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to create file tag
def create_file_tag(section_ts_start, section_ts_end, section_ens=None, section_name=None,
                    time_format='%Y%m%d%H%M', tag_sep=':'):

    if (section_ens is not None) and (section_name is None):
        section_tag = section_ts_start.strftime(time_format) + '_' + section_ts_end.strftime(time_format) + \
                      tag_sep + section_ens
    elif (section_name is not None) and (section_ens is None):
        section_tag = section_ts_start.strftime(time_format) + '_' + section_ts_end.strftime(time_format) + \
                      tag_sep + section_name
    elif (section_name is None) and (section_ens is None):
        section_tag = section_ts_start.strftime(time_format) + '_' + section_ts_end.strftime(time_format)
    else:
        log_stream.error(' ===> File tag is not correctly defined')
        raise NotImplementedError('Case not implemented yet')

    return section_tag
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to create obj hydro
def create_obj_hydro(file_index, file_list_key, file_list_data):
    file_data = {}
    for file_step_key, file_step_data in zip(file_list_key, file_list_data):
        file_data[file_step_key] = file_step_data
    file_dframe = pd.DataFrame(index=file_index, data=file_data)
    return file_dframe
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to analyze obj hydro
def analyze_obj_hydro(file_obj_in, file_source='sim', file_method='max',
                      file_tag_discharge='discharge', file_tag_type='type'):
    file_obj_out = {}
    for file_key, file_dframe_in in file_obj_in.items():

        if file_dframe_in is not None:
            file_time = list(file_dframe_in.index)
            file_type = [file_source] * file_time.__len__()

            if file_method is not None:
                if file_method == 'max':
                    file_values = list(file_dframe_in.max(axis=1).values)
                elif file_method == 'mean':
                    file_values = list(file_dframe_in.mean(axis=1).values)
                else:
                    log_stream.error(' ===> Analyze method for file values is not correctly defined')
                    raise NotImplementedError('Case not implemented yet')
            else:
                file_values = list(file_dframe_in.values[:, 0])

            file_dframe_out = create_obj_hydro(file_time,
                                               [file_tag_discharge, file_tag_type], [file_values, file_type])

        else:
            file_dframe_out = None

        file_obj_out[file_key] = file_dframe_out

    return file_obj_out

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to parse filename in parts
def parse_file_parts(file_name, file_type='sim'):

    file_parts = re.findall(r'\d+', file_name)

    file_root, file_extension = os.path.splitext(file_name)

    if file_parts.__len__() == 3:

        if file_type == 'sim':
            # probabilistic simulated file(s)
            file_part_datetime_start = datetime.strptime(file_parts[0], "%y%j%H%M")
            file_part_datetime_end = datetime.strptime(file_parts[1][:-2], "%y%j%H%M")
            file_part_mask = file_parts[1][-2:]
            file_part_n_ens = file_parts[2]
        elif file_type == 'obs':
            file_part_datetime_start = datetime.strptime(file_parts[0], "%Y%m%d%H%M")
            file_part_datetime_end = datetime.strptime(file_parts[1], "%Y%m%d%H%M")
            file_part_mask = None
            file_part_n_ens = None
        else:
            log_stream.error(' ===> Parser of filename ' + file_name + ' failed for unknown type of file parts equal to 3')
            raise NotImplementedError('Case not implemented yet')

    elif file_parts.__len__() == 2:

        if file_type == 'sim':
            # deterministic simulated file(s)
            file_part_datetime_start = datetime.strptime(file_parts[0], "%y%j%H%M")
            file_part_datetime_end = datetime.strptime(file_parts[1][:-2], "%y%j%H%M")
            file_part_mask = file_parts[1][-2:]
            file_part_n_ens = None
        elif file_type == 'obs':
            if file_extension == '.mat':
                file_part_datetime_start = datetime.strptime(file_parts[0], "%y%j%H%M")
                file_part_datetime_end = datetime.strptime(file_parts[1][:-2], "%y%j%H%M")
                file_part_mask = file_parts[1][-2:]
                file_part_n_ens = None
            elif file_extension == '.txt':
                file_part_datetime_start = datetime.strptime(file_parts[0], "%Y%m%d%H%M")
                file_part_datetime_end = datetime.strptime(file_parts[1], "%Y%m%d%H%M")
                file_part_mask = None
                file_part_n_ens = None
            else:
                log_stream.error(' ===> Parser of filename ' + file_name + ' failed for unknown file format')
                raise NotImplementedError('Case not implemented yet')
        else:
            log_stream.error(' ===> Parser of filename ' + file_name + ' failed for unknown file type')
            raise NotImplementedError('Case not implemented yet')
    else:
        log_stream.error(' ===> Parser of filename ' + file_name + ' failed for unknown format')
        raise NotImplementedError('Case not implemented yet')

    file_part_timestamp_start = pd.Timestamp(file_part_datetime_start)
    file_part_timestamp_end = pd.Timestamp(file_part_datetime_end)

    return file_part_timestamp_start, file_part_timestamp_end, file_part_mask, file_part_n_ens
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read hydro simulated file in ascii format
def read_file_hydro_sim(section_name, file_name, column_time_idx=0, method_data_filling='ffill'):

    file_data = pd.read_table(file_name)
    file_data = file_data.loc[:, ~file_data.columns.str.match("Unnamed")]

    file_columns = list(file_data.columns)
    if isinstance(file_columns, list):
        if file_columns.__len__() == 1:
            file_cols_tmp = file_columns[0].split(' ')
            file_cols_filtered = list(filter(None, file_cols_tmp))
        elif file_columns.__len__() > 1:
            file_cols_filtered = deepcopy(file_columns)
        else:
            log_stream.error(' ===> File columns dimension is not allowed')
            raise NotImplementedError('Case not implemented yet')
    else:
        log_stream.error(' ===> File column format is not supported')
        raise NotImplementedError('Case not implemented yet')

    if section_name in file_cols_filtered:

        column_section_idx = file_cols_filtered.index(section_name)
        file_data_table = list(file_data.values)

        section_period = []
        section_data = []
        for file_data_row in file_data_table:

            file_row = list(file_data_row)
            if file_columns.__len__() == 1:
                file_data_parts = file_data_row[0].split(' ')
                file_data_parts = list(filter(None, file_data_parts))
            elif file_columns.__len__() > 1:
                file_data_parts = deepcopy(file_row)
            else:
                log_stream.error(' ===> File row dimension is not allowed')
                raise NotImplementedError('Case not implemented yet')

            section_time_string = str(int(file_data_parts[column_time_idx]))
            section_point = float(file_data_parts[column_section_idx])

            section_time_stamp = pd.Timestamp(section_time_string)

            section_period.append(section_time_stamp)
            section_data.append(section_point)

        section_series = pd.Series(index=section_period, data=section_data)

        section_series = fill_obj_hydro(
            obj_hydro=section_series, obj_method_filling=method_data_filling)

    else:
        section_series = np.nan

    return section_series

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read hydro observed file in mat and txt format
def read_file_hydro_obs(section_name, file_name,
                        column_time_in='a1sDateVet', column_discharge_in='a1dQOss', column_level_in='a1dLivOssMean',
                        column_time_out='time', column_discharge_out='discharge', column_level_out='water_level',
                        method_data_filling='ffill'):

    if file_name.endswith('.mat'):
        file_data = loadmat(file_name)

        if column_time_in in list(file_data.keys()):
            file_time = file_data[column_time_in]
        else:
            log_stream.error(' ===> File column "' + column_time_in + '" observed not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

        if column_discharge_in in list(file_data.keys()):
            file_discharge = file_data[column_discharge_in]
        else:
            log_stream.error(' ===> File column "' + column_discharge_in + '" observed not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

        if column_level_in is not None:
            if column_level_in in list(file_data.keys()):
                file_water_level = file_data[column_level_in]
            else:
                log_stream.warning(' ===> File column "' + column_level_in + '" observed not available in the datasets')
                file_water_level = np.zeros(shape=(file_discharge.shape[0], 1))
                file_water_level[:, :] = -9999.0
        else:
            log_stream.warning(' ===> File column for water level observed variable is undefined')
            file_water_level = np.zeros(shape=(file_discharge.shape[0], 1))
            file_water_level[:, :] = -9999.0

        time_list = file_time[:, 0].tolist()
        discharge_list = file_discharge[:, 0].tolist()
        water_level_list = file_water_level[:, 0].tolist()

        section_period = []
        section_data_discharge = []
        section_data_water_level = []
        for time_step, discharge_step, water_level_step in zip(time_list, discharge_list, water_level_list):

            section_time = pd.Timestamp(pd.Timestamp(str(time_step[0])))
            section_point_discharge = float(discharge_step)
            section_point_water_level = float(water_level_step)

            section_period.append(section_time)
            section_data_discharge.append(section_point_discharge)
            section_data_water_level.append(section_point_water_level)

        section_series_discharge = pd.Series(index=section_period, data=section_data_discharge)
        section_series_water_level = pd.Series(index=section_period, data=section_data_water_level)

        section_series_discharge = fill_obj_hydro(
            obj_hydro=section_series_discharge, obj_method_filling=method_data_filling)

    elif file_name.endswith('.txt'):

        file_table = pd.read_table(file_name, sep=' ')

        if column_time_in in list(file_table.columns):
            file_time = file_table[column_time_in].values
        else:
            log_stream.error(' ===> File column "' + column_time_in + '" observed not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

        if column_discharge_in in list(file_table.columns):
            file_discharge = file_table[column_discharge_in].values
        else:
            log_stream.error(' ===> File column "' + column_discharge_in + '" observed not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

        if column_level_in is not None:
            if column_level_in in list(file_table.columns):
                file_water_level = file_table[column_level_in].values
            else:
                log_stream.warning(' ===> File column "' + column_level_in + '" observed not available in the datasets')
                file_water_level = np.zeros(shape=file_discharge.shape[0])
                file_water_level[:] = -9999.0
        else:
            log_stream.warning(' ===> File column for water level observed variable is undefined')
            file_water_level = np.zeros(shape=file_discharge.shape[0])
            file_water_level[:] = -9999.0

        time_list = file_time.tolist()
        discharge_list = file_discharge.tolist()
        water_level_list = file_water_level.tolist()

        section_period = []
        section_data_discharge = []
        section_data_water_level = []
        for time_step, discharge_step, water_level_step in zip(time_list, discharge_list, water_level_list):

            section_time = pd.Timestamp(pd.Timestamp(str(time_step)))
            section_point_discharge = float(discharge_step)
            section_point_water_level = float(water_level_step)

            section_period.append(section_time)
            section_data_discharge.append(section_point_discharge)
            section_data_water_level.append(section_point_water_level)

        section_series_discharge = pd.Series(index=section_period, data=section_data_discharge)
        section_series_water_level = pd.Series(index=section_period, data=section_data_water_level)

        section_series_discharge = fill_obj_hydro(
            obj_hydro=section_series_discharge, obj_method_filling=method_data_filling)

    else:
        log_stream.error(' ===> File "' + file_name + '" unsupported format')
        raise NotImplementedError('Case not implemented yet')

    return section_series_discharge, section_series_water_level

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to fill obj hydro
def fill_obj_hydro(obj_hydro, obj_method_filling='ffill', obj_limit_filling=10,
                   obj_value_min=0.0, obj_value_max=None, obj_nodata=-9999.0):

    if obj_value_min is not None:
        obj_hydro[obj_hydro < obj_value_min] = np.nan
    if obj_value_max is not None:
        obj_hydro[obj_hydro > obj_value_max] = np.nan

    time_valid_first = obj_hydro.first_valid_index()
    time_valid_last = obj_hydro.last_valid_index()

    if obj_method_filling is not None:
        if obj_method_filling == 'ffill':
            obj_hydro_filled = obj_hydro.ffill(axis='rows', limit=obj_limit_filling)
        elif obj_method_filling == 'bfill':
            obj_hydro_filled = obj_hydro.bfill(axis='rows', limit=obj_limit_filling)
        elif obj_method_filling == 'interpolate':
            obj_hydro_filled = obj_hydro.interpolate(method='values', limit=obj_limit_filling)
        else:
            log_stream.error(' ===> Method to fill data "' + obj_method_filling + '" is not suppoerted')
            raise NotImplementedError('Case not implemented yet')
    else:
        obj_hydro_filled = deepcopy(obj_hydro)

    obj_hydro_filled = obj_hydro_filled.fillna(obj_nodata)
    obj_hydro_filtered = obj_hydro_filled[time_valid_first:time_valid_last]

    return obj_hydro_filtered

# -------------------------------------------------------------------------------------
