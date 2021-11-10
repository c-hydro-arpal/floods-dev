# -------------------------------------------------------------------------------------
# Libraries
import logging
import os
import re

from scipy.io import loadmat
from datetime import datetime

import numpy as np
import pandas as pd
# -------------------------------------------------------------------------------------


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
        logging.error(' ===> File tag is not correctly defined')
        raise NotImplementedError('Case not implemented yet')

    return section_tag
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to parse filename in parts
def parse_file_parts(file_name, file_type='sim'):

    file_parts = re.findall(r'\d+', file_name)

    if file_parts.__len__() == 3:
        if file_type == 'sim':
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
            logging.error(' ===> Parser of filename ' + file_name + ' fails for unknown type of file parts equal to 3')
            raise NotImplementedError('Case not implemented yet')

    elif file_parts.__len__() == 2:
        file_part_datetime_start = datetime.strptime(file_parts[0], "%y%j%H%M")
        file_part_datetime_end = datetime.strptime(file_parts[1][:-2], "%y%j%H%M")
        file_part_mask = file_parts[1][-2:]
        file_part_n_ens = None
    else:
        logging.error(' ===> Parser of filename ' + file_name + ' fails for unknown format')
        raise NotImplementedError('Case not implemented yet')

    file_part_timestamp_start = pd.Timestamp(file_part_datetime_start)
    file_part_timestamp_end = pd.Timestamp(file_part_datetime_end)

    return file_part_timestamp_start, file_part_timestamp_end, file_part_mask, file_part_n_ens
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read hydro simulated file in ascii format
def read_file_hydro_sim(section_name, file_name, column_time_idx=0):

    file_data = pd.read_table(file_name)

    file_cols_tmp = list(file_data.columns)[0].split(' ')
    file_cols_filtered = list(filter(None, file_cols_tmp))

    if section_name in file_cols_filtered:

        column_section_idx = file_cols_filtered.index(section_name)
        file_data_table = list(file_data.values)

        section_period = []
        section_data = []
        for file_data_row in file_data_table:
            file_data_parts = list(file_data_row)[0].split(' ')
            file_data_parts = list(filter(None, file_data_parts))

            section_time = pd.Timestamp(file_data_parts[column_time_idx])
            section_point = float(file_data_parts[column_section_idx])

            section_period.append(section_time)
            section_data.append(section_point)

        section_series = pd.Series(index=section_period, data=section_data)

    else:
        section_series = np.nan

    return section_series

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read hydro observed file in mat format
def read_file_hydro_obs(section_name, file_name,
                        column_time_in='a1sDateVet', column_discharge_in='a1dQOss', column_level_in='a1dLivOssMean',
                        column_time_out='time', column_discharge_out='discharge', column_level_out='water_level'):

    if file_name.endswith('.mat'):
        file_data = loadmat(file_name)

        if column_time_in in list(file_data.keys()):
            file_time = file_data[column_time_in]
        else:
            logging.error(' ===> File column "' + column_time_in + '" not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

        if column_discharge_in in list(file_data.keys()):
            file_discharge = file_data[column_discharge_in]
        else:
            logging.error(' ===> File column "' + column_discharge_in + '" not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

        if column_level_in in list(file_data.keys()):
            file_water_level = file_data[column_level_in]
        else:
            logging.error(' ===> File column "' + column_level_in + '" not available in the datasets')
            raise IOError('Check your input file "' + file_name + '" to control the available fields')

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

        # section_dframe = pd.DataFrame(index=section_period,
        #                               data={column_discharge_out: section_data_discharge,
        #                                     column_level_out: section_data_water_level})

        section_series_discharge = pd.Series(index=section_period, data=section_data_discharge)
        section_series_water_level = pd.Series(index=section_period, data=section_data_water_level)

    else:
        logging.error(' ===> File "' + file_name + '" unsupported format')
        raise NotImplementedError('Case not implemented yet')

    return section_series_discharge, section_series_water_level

# -------------------------------------------------------------------------------------
