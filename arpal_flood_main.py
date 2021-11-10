#!/usr/bin/python3
"""
ARPAL Processing Tool - FLOODS SCENARIO

__date__ = '20211005'
__version__ = '1.7.0'
__author__ =
        'Fabio Delogu (fabio.delogu@cimafoundation.org',
        'Flavio Pignone (flavio.pignone@cimafoundation.org',
        'Rocco Masi (rocco.masi@cimafoundation.org',
        'Lorenzo Campo (lorenzo.campo@cimafoundation.org',
        'Francesco Silvestro (francesco.silvestro@cimafoundation.org'

__library__ = 'ARPAL'

General command line:
python3 arpal_flood_main.py -settings_file configuration.json -time "YYYY-MM-DD HH:MM"

Version(s):
20211005 (1.7.0) --> Generic release fpr correcting bugs and managing the empty datasets for observed/modelled discharges
20210515 (1.6.0) --> Generic release for updating tools and modules
20201214 (1.5.0) --> Pre-operational release
20200522 (1.0.0) --> Beta release
"""

# -------------------------------------------------------------------------------------
# Complete library
import logging
import time
import os

from driver_data_io_geo import DriverGeo
from driver_data_io_source import DriverDischarge
from driver_data_io_destination import DriverScenario

from argparse import ArgumentParser

# from lib_utils_geo import drainage_area
from lib_utils_io import read_file_json
from lib_utils_system import make_folder
from lib_utils_time import set_time
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_version = '1.7.0'
alg_release = '2021-10-05'
alg_name = 'FLOODS SCENARIO'
# Algorithm parameter(s)
time_format = '%Y-%m-%d %H:%M'
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Script Main
def main():

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Set algorithm logging
    make_folder(data_settings['log']['folder_name'])
    set_logging(logger_file=os.path.join(data_settings['log']['folder_name'],
                                         data_settings['log']['file_name']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> START ... ')
    logging.info(' ')

    # Time algorithm information
    start_time = time.time()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Organize time run
    time_now, time_run, time_range = set_time(
        time_run_args=alg_time,
        time_run_file=data_settings['time']['time_now'],
        time_format=time_format,
        time_period=data_settings['time']['time_period'],
        time_frequency=data_settings['time']['time_frequency'],
        time_rounding=data_settings['time']['time_rounding']
    )
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Geographical datasets
    driver_data_geo = DriverGeo(
        src_dict=data_settings['data']['static']['source'],
        dst_dict=data_settings['data']['static']['destination'],
        alg_ancillary=data_settings['algorithm']['ancillary'],
        alg_template_tags=data_settings['algorithm']['template'],
        flag_cleaning_geo=data_settings['algorithm']['flags']['cleaning_geo_data'])
    geo_data_collection = driver_data_geo.organize_geo()
    # -------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # Iterate over time range
    for time_step in time_range:

        # -------------------------------------------------------------------------------------
        # Discharge datasets
        driver_data_source_discharge = DriverDischarge(
            time_now=time_now,
            time_run=time_step,
            geo_data_collection=geo_data_collection,
            src_dict=data_settings['data']['dynamic']['source'],
            ancillary_dict=data_settings['data']['dynamic']['ancillary'],
            alg_ancillary=data_settings['algorithm']['ancillary'],
            alg_template_tags=data_settings['algorithm']['template'],
            flag_cleaning_ancillary=data_settings['algorithm']['flags']['cleaning_dynamic_data_ancillary_discharge'])
        discharge_data_collection = driver_data_source_discharge.organize_discharge()

        # Scenario datasets
        driver_data_destination_scenario = DriverScenario(
            time_now=time_now,
            time_run=time_step,
            discharge_data_collection=discharge_data_collection,
            geo_data_collection=geo_data_collection,
            src_dict=data_settings['data']['static']['source'],
            ancillary_dict=data_settings['data']['dynamic']['ancillary'],
            dst_dict=data_settings['data']['dynamic']['destination'],
            alg_ancillary=data_settings['algorithm']['ancillary'],
            alg_template_tags=data_settings['algorithm']['template'],
            flag_cleaning_ancillary=data_settings['algorithm']['flags']['cleaning_dynamic_data_ancillary_scenario'],
            flag_cleaning_scenario=data_settings['algorithm']['flags']['cleaning_scenario_data'])
        scenario_info_collection = driver_data_destination_scenario.organize_scenario_datasets()
        scenario_map_collection = driver_data_destination_scenario.compute_scenario_map(scenario_info_collection)
        driver_data_destination_scenario.dump_scenario_map(scenario_map_collection, scenario_info_collection)
        # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    time_elapsed = round(time.time() - start_time, 1)

    logging.info(' ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME ELAPSED: ' + str(time_elapsed) + ' seconds')
    logging.info(' ==> ... END')
    logging.info(' ==> Bye, Bye')
    logging.info(' ============================================================================ ')

    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time', action="store", dest="alg_time")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time:
        alg_time = parser_values.alg_time
    else:
        alg_time = None

    return alg_settings, alg_time

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.DEBUG)

    # Open logging basic configuration
    logging.basicConfig(level=logging.DEBUG, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.DEBUG)
    logger_handle_2.setLevel(logging.DEBUG)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)

    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)

# -------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == '__main__':
    main()
# ----------------------------------------------------------------------------
