"""
Library Features:

Name:          lib_utils_section
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220208'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import os
import json

import pandas as pd

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
#######################################################################################


# -------------------------------------------------------------------------------------
# Section default variables map
obj_section_fields_default = {
    "version": "default",
    "fields_map": {
        'CodiceOrig': 'section_code', 'Mask': 'section_mask_reference',
        'codice': 'section_name', 'Stazione': 'section_point_station',
        'Bacino': 'section_catchment_name', 'Provincia': 'section_province',
        'Y_Grid': 'section_grid_y', 'X_Grid': 'section_grid_x',
        'Area drift': 'section_drainage_area', 'Classe_bac': 'section_catchment_group',
        'num_mask': 'section_mask_n', 'Quota': 'section_altitude', 'Sc_defluss': 'section_rating_curve',
        'Q 2.9': 'section_q_2.9', 'Q 5': 'section_q_5', 'Q 10': 'section_q_10', 'Q 20': 'section_q_20',
        'Q 30': 'section_q_30', 'Q 50': 'section_q_50', 'Q 100': 'section_q_100', 'Q 200': 'section_q_200',
        'Lat': 'section_latitude', 'Lon': 'section_longitude',
        'section_name': '__section_name__', 'river_station': 'section_hydrometer_code',
        'undefined_code': '__undefined_code__'
    },
    "fields_excluded": ['__section_name__', '__undefined_code__']
}
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to map section information
def map_info_section(section_data_in, section_fields_map=None, section_fields_excluded=None):

    if section_fields_map is not None:
        section_data_tmp = section_data_in.rename(columns=section_fields_map)
    else:
        log_stream.error(' ===> Section map fields obj is not defined')
        raise RuntimeError('Obj fields map must be defined in the procedure')

    if section_fields_excluded is not None:
        section_data_out = section_data_tmp.drop(columns=section_fields_excluded)
    else:
        log_stream.error(' ===> Section excluded fields obj is not defined')
        raise RuntimeError('Obj fields excluded must be defined in the procedure')

    return section_data_out

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to merge section information
def merge_info_section(section_data, other_data, section_data_ref='CodiceOrig', other_data_ref='section_name'):
    section_data_merged = section_data.merge(other_data, how='left', left_on=section_data_ref, right_on=other_data_ref)
    return section_data_merged
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read file (or default settings) to convert fields of section file
def read_file_fields(file_name, tag_fields_variable='fields_map', tag_fields_excluded='fields_excluded'):

    if (file_name is not None) and (os.path.exists(file_name)):
        with open(file_name) as file_handle:
            file_data = json.load(file_handle)
    else:
        file_data = obj_section_fields_default.copy()

    fields_map = file_data[tag_fields_variable]
    fields_excluded = file_data[tag_fields_excluded]

    return fields_map, fields_excluded
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read river station look-up table in ascii format
def read_file_river_station_lut(file_name, file_cols_name=None):

    if file_cols_name is None:
        file_cols_name = ['section_name', 'river_station', 'undefined_code']

    file_lut = pd.read_table(file_name, names=file_cols_name, sep=' ')
    file_lut.reset_index()

    return file_lut

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read section file in csv format
def read_file_section(file_name, file_sep=';'):
    file_data = pd.read_csv(file_name, sep=file_sep)
    return file_data
# -------------------------------------------------------------------------------------
