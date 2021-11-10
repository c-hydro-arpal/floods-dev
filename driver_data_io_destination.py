"""
Class Features

Name:          driver_data_io_destination
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20200515'
Version:       '1.0.0'
"""

######################################################################################
# Library
import logging
import os
import numpy as np

from copy import deepcopy

import pandas as pd

from lib_utils_geo import read_file_geo
from lib_utils_hazard import read_file_hazard
from lib_utils_io import read_file_json, read_obj, write_obj, write_file_tif
from lib_utils_system import fill_tags2string, make_folder
from lib_utils_generic import get_dict_value
from lib_utils_plot import save_file_tiff, save_file_png, save_file_json

# Debug
import matplotlib.pylab as plt

# Script definition
time_string_format = "%Y-%m-%d %H:%M"
######################################################################################


# -------------------------------------------------------------------------------------
# Class DriverScenario
class DriverScenario:

    # -------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, time_now, time_run, discharge_data_collection, geo_data_collection,
                 src_dict, ancillary_dict, dst_dict,
                 alg_ancillary=None, alg_template_tags=None,
                 flag_telemac_data='telemac_data', flag_hazard_data='hazard_data',
                 flag_scenario_data='scenario_data',
                 flag_scenario_plot_tiff='scenario_plot_tiff', flag_scenario_plot_png='scenario_plot_png',
                 flag_cleaning_ancillary=True, flag_cleaning_scenario=True):

        self.time_now = time_now
        self.time_run = time_run

        self.discharge_data_collection = discharge_data_collection
        self.geo_data_collection = geo_data_collection

        self.flag_telemac_data = flag_telemac_data
        self.flag_hazard_data = flag_hazard_data
        self.flag_scenario_data = flag_scenario_data
        self.flag_scenario_plot_tiff = flag_scenario_plot_tiff
        self.flag_scenario_plot_png = flag_scenario_plot_png

        self.alg_ancillary = alg_ancillary
        self.tr_min = alg_ancillary['tr_min']
        self.tr_max = alg_ancillary['tr_max']
        self.tr_freq = alg_ancillary['tr_freq']

        self.scenario_analysis = alg_ancillary['scenario_analysis']
        self.scenario_method = alg_ancillary['scenario_method']

        self.alg_template_tags = alg_template_tags
        self.file_name_tag = 'file_name'
        self.folder_name_tag = 'folder_name'
        self.save_status_tag = 'save_status'

        self.domain_name_list = self.alg_ancillary['domain_name']

        self.folder_name_telemac = src_dict[self.flag_telemac_data][self.folder_name_tag]
        self.file_name_telemac = src_dict[self.flag_telemac_data][self.file_name_tag]

        self.folder_name_hazard = src_dict[self.flag_hazard_data][self.folder_name_tag]
        self.file_name_hazard = src_dict[self.flag_hazard_data][self.file_name_tag]

        self.folder_name_scenario_ancillary = ancillary_dict[self.flag_scenario_data][self.folder_name_tag]
        self.file_name_scenario_ancillary = ancillary_dict[self.flag_scenario_data][self.file_name_tag]

        self.format_tr = '{:03d}'
        self.scenario_tr = self.define_tr_scenario(self.tr_min, self.tr_max, self.tr_freq)

        self.folder_name_scenario_data = dst_dict[self.flag_scenario_data][self.folder_name_tag]
        self.file_name_scenario_data = dst_dict[self.flag_scenario_data][self.file_name_tag]
        self.save_status_scenario_data = dst_dict[self.flag_scenario_data][self.save_status_tag]
        self.folder_name_scenario_plot_tiff = dst_dict[self.flag_scenario_plot_tiff][self.folder_name_tag]
        self.file_name_scenario_plot_tiff = dst_dict[self.flag_scenario_plot_tiff][self.file_name_tag]
        self.save_status_scenario_plot_tiff = dst_dict[self.flag_scenario_plot_tiff][self.save_status_tag]
        self.folder_name_scenario_plot_png = dst_dict[self.flag_scenario_plot_png][self.folder_name_tag]
        self.file_name_scenario_plot_png = dst_dict[self.flag_scenario_plot_png][self.file_name_tag]
        self.save_status_scenario_plot_png = dst_dict[self.flag_scenario_plot_png][self.save_status_tag]

        self.flag_cleaning_ancillary = flag_cleaning_ancillary
        self.flag_cleaning_scenario = flag_cleaning_scenario

        scenario_description_collection = {}
        for domain_name_step in self.domain_name_list:
            domain_description_list = get_dict_value(geo_data_collection[domain_name_step], 'description', [])
            scenario_description_collection[domain_name_step] = domain_description_list
        self.scenario_description_collection = scenario_description_collection

        self.domain_discharge_index_tag = 'discharge_idx'
        self.domain_grid_x_tag = 'grid_x_grid'
        self.domain_grid_y_tag = 'grid_y_grid'
        self.domain_sections_db_tag = 'domain_sections_db'

        self.domain_scenario_index_tag = 'scenario_idx'
        self.domain_scenario_discharge_tag = 'discharge_value'
        self.domain_scenario_time_tag = 'time'
        self.domain_scenario_n_tag = 'scenario_n'

        self.domain_scenario_area_tag = "mappa_aree_new"
        self.domain_scenario_grid_x_tag = "new_x"
        self.domain_scenario_grid_y_tag = "new_y"

        self.domain_scenario_hazard_tag = 'mappa_h'

        self.domain_name_tag = 'domain_name'

        self.scale_factor_hazard = 1000
        # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to define hazard file
    def define_file_hazard(self, folder_name_raw, file_name_raw, domain_name, section_tr):

        template_tags = self.alg_template_tags

        template_values_step = {'domain_name': domain_name, 'tr': self.format_tr.format(section_tr)}

        folder_name_def = fill_tags2string(folder_name_raw, template_tags, template_values_step)
        file_name_def = fill_tags2string(file_name_raw, template_tags, template_values_step)
        path_name_def = os.path.join(folder_name_def, file_name_def)

        return path_name_def

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to define scenarios tr
    def define_tr_scenario(self, tr_min, tr_max, tr_freq=1):
        scenario_tr_raw = np.arange(tr_min, tr_max + 1, tr_freq).tolist()
        scenario_tr_def = []
        for scenario_step in scenario_tr_raw:
            scenario_tmp = self.format_tr.format(scenario_step)
            scenario_tr_def.append(scenario_tmp)
        return scenario_tr_def
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to compute tr for evaluating scenario
    @staticmethod
    def compute_scenario_tr(section_discharge_idx, section_discharge_values):

        if not isinstance(section_discharge_values, list):
            section_discharge_values = [section_discharge_values]

        if section_discharge_idx > 0.0:

            section_scenario_trs = []
            for section_discharge_value in section_discharge_values:

                if section_discharge_value >= 0.0:
                    section_scenario_tr = np.round(np.exp(
                        (section_discharge_idx * 0.5239 + section_discharge_value) / (section_discharge_idx * 1.0433)))
                    section_scenario_tr = int(section_scenario_tr)
                else:
                    section_scenario_tr = np.nan

                section_scenario_trs.append(section_scenario_tr)
        else:
            section_scenario_trs = [np.nan] * section_discharge_values.__len__()

        return section_scenario_trs
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to compute discharge for evaluating scenario
    @staticmethod
    def compute_scenario_discharge(dframe_discharge, analysis_freq=None, analysis_method='max'):

        if analysis_freq is None:
            if analysis_method == 'max':
                reference_value = list(dframe_discharge.idxmax().index)
                time_value = list(dframe_discharge.idxmax())
                occurrence_value = list(dframe_discharge.idxmax().index).__len__()
                discharge_value = list(dframe_discharge.max())
            else:
                logging.error(' ===> Method to compute discharge for evaluating scenario is not defined')
                raise NotImplemented('Method not implemented yet')

        elif analysis_freq == 'ALL':

            reference_value = []
            time_value = []
            occurrence_value = []
            discharge_value = []
            for id_step, (time_step, value_step) in enumerate(zip(dframe_discharge.index, dframe_discharge.values)):

                value_step = value_step[0]
                if not np.isnan(value_step):
                    reference_value.append(id_step)
                    time_value.append(time_step)
                    discharge_value.append(value_step)
                    occurrence_value.append(1)
        else:
            logging.error(' ===> Frequency to compute discharge for evaluating scenario is not defined')
            raise NotImplemented('Method not implemented yet')

        return reference_value, time_value, discharge_value, occurrence_value
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to define hazard file
    def define_file_scenario(self, time_run, folder_name_raw, file_name_raw, domain_name, time_step=None):

        template_tags = self.alg_template_tags

        if time_step is None:
            time_step = time_run

        template_values_step = {'domain_name': domain_name,
                                'destination_sub_path_time_scenario': time_run,
                                'ancillary_sub_path_time_scenario': time_run,
                                'destination_datetime_scenario': time_step,
                                'ancillary_datetime_scenario': time_step
                                }

        folder_name_def = fill_tags2string(folder_name_raw, template_tags, template_values_step)
        file_name_def = fill_tags2string(file_name_raw, template_tags, template_values_step)
        path_name_def = os.path.join(folder_name_def, file_name_def)

        return path_name_def

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to dump scenario map
    def dump_scenario_map(self, scenario_map_collection, scenario_info_collection):

        time_run = self.time_run
        time_now = self.time_now

        time_now_string = time_now.strftime(time_string_format)
        geo_data_collection = self.geo_data_collection
        scenario_description_collection = self.scenario_description_collection

        logging.info(' ---> Dump scenario maps [' + time_run.strftime(time_string_format) + '] ... ')

        for domain_name_step in self.domain_name_list:

            logging.info(' ----> Domain "' + domain_name_step + '" ... ')

            domain_geo_collection = geo_data_collection[domain_name_step]
            domain_info_collection = scenario_info_collection[domain_name_step]
            domain_map_collection = scenario_map_collection[domain_name_step]
            domain_description_collection = scenario_description_collection[domain_name_step]

            if not domain_map_collection:
                domain_map_collection = None

            if domain_map_collection is not None:

                for domain_map_time, domain_map_file_ancillary in domain_map_collection.items():

                    logging.info(' -----> Time step "' + domain_map_time.strftime(time_string_format) + '" ... ')

                    logging.info(' ------> Prepare file data ... ')
                    domain_map_data = read_obj(domain_map_file_ancillary)

                    file_path_scenario_data = self.define_file_scenario(
                        time_now, self.folder_name_scenario_data, self.file_name_scenario_data,
                        domain_name_step, time_step=domain_map_time)
                    file_path_scenario_plot_tiff = self.define_file_scenario(
                        time_now, self.folder_name_scenario_plot_tiff, self.file_name_scenario_plot_tiff,
                        domain_name_step, time_step=domain_map_time)
                    file_path_scenario_plot_png = self.define_file_scenario(
                        time_now, self.folder_name_scenario_plot_png, self.file_name_scenario_plot_png,
                        domain_name_step, time_step=domain_map_time)

                    if self.flag_cleaning_scenario:
                        if os.path.exists(file_path_scenario_data):
                            os.remove(file_path_scenario_data)
                        if os.path.exists(file_path_scenario_plot_tiff):
                            os.remove(file_path_scenario_plot_tiff)
                        if os.path.exists(file_path_scenario_plot_png):
                            os.remove(file_path_scenario_plot_png)

                    domain_geo_data = domain_geo_collection[self.domain_scenario_area_tag]
                    domain_geo_x = domain_geo_collection[self.domain_scenario_grid_x_tag]
                    domain_geo_y = domain_geo_collection[self.domain_scenario_grid_y_tag]

                    time_step_string = domain_map_time.strftime(time_string_format)

                    section_info_collection = {}
                    for domain_info_key, domain_info_fields in domain_info_collection.items():
                        if domain_info_fields is not None:
                            if domain_info_key in domain_description_collection:

                                domain_info_dframe = pd.DataFrame(domain_info_fields, index=domain_info_fields['time'])
                                section_info_fields = domain_info_dframe[domain_info_dframe.index.isin([domain_map_time])].to_dict('r')[0]

                                for section_info_key, section_info_value in section_info_fields.items():
                                    if isinstance(section_info_value, pd.Timestamp):
                                        section_tmp_value = section_info_value.strftime(time_string_format)
                                        section_info_fields[section_info_key] = section_tmp_value

                                section_info_collection[domain_info_key] = section_info_fields

                    section_info_collection['scenario_name'] = domain_name_step
                    section_info_collection['scenario_time_now'] = time_now_string
                    section_info_collection['scenario_time_step'] = time_step_string

                    logging.info(' ------> Prepare file data ... DONE')

                    # Save information in json file
                    folder_name_scenario_data, file_name_scenario_data = os.path.split(file_path_scenario_data)
                    make_folder(folder_name_scenario_data)

                    logging.info(' ------> Save file json ' + file_name_scenario_data + ' ... ')
                    if self.save_status_scenario_data:
                        if not os.path.exists(file_path_scenario_data):
                            save_file_json(file_path_scenario_data, section_info_collection)
                            logging.info(' ------> Save file json ' + file_name_scenario_data +
                                         ' ... DONE')
                        else:
                            logging.info(' ------> Save file json ' + file_name_scenario_data +
                                         ' ... PREVIOUSLY SAVED')
                    else:
                        logging.info(' ------> Save file json ' + file_name_scenario_data +
                                     ' ... SKIPPED. Save method is deactivated')

                    # Save information in png file
                    folder_name_scenario_plot_png, file_name_scenario_plot_png = os.path.split(
                        file_path_scenario_plot_png)
                    make_folder(folder_name_scenario_plot_png)

                    logging.info(' ------> Save file png ' + file_name_scenario_plot_png + ' ... ')
                    if self.save_status_scenario_plot_png:
                        if not os.path.exists(file_path_scenario_plot_png):
                            save_file_png(file_path_scenario_plot_png,
                                          domain_map_data, domain_geo_x, domain_geo_y,
                                          scenario_name=domain_name_step,
                                          scenario_time_now_string=time_now_string,
                                          scenario_time_step_string=time_step_string,
                                          fig_color_map_type=None, fig_dpi=150)
                            logging.info(' ------> Save file png ' + file_name_scenario_plot_png +
                                         ' ... DONE')
                        else:
                            logging.info(' ------> Save file png ' + file_name_scenario_plot_png +
                                         ' ... PREVIOUSLY SAVED')
                    else:
                        logging.info(' ------> Save file png ' + file_name_scenario_plot_png +
                                     ' ... SKIPPED. Save method is deactivated')

                    # Save information in tiff file
                    folder_name_scenario_plot_tiff, file_name_scenario_plot_tiff = os.path.split(
                        file_path_scenario_plot_tiff)
                    make_folder(folder_name_scenario_plot_tiff)

                    logging.info(' ------> Save file tiff ' + file_name_scenario_plot_tiff + ' ... ')
                    if self.save_status_scenario_plot_tiff:
                        if not os.path.exists(file_path_scenario_plot_tiff):
                            save_file_tiff(file_path_scenario_plot_tiff,
                                           domain_map_data, domain_geo_x, domain_geo_y,
                                           file_epsg_code='EPSG:32632')
                            logging.info(' ------> Save file tiff ' + file_name_scenario_plot_tiff +
                                         ' ... DONE')
                        else:
                            logging.info(' ------> Save file tiff ' + file_name_scenario_plot_tiff +
                                         ' ... PREVIOUSLY SAVED')
                    else:
                        logging.info(' ------> Save file tiff ' + file_name_scenario_plot_tiff +
                                     ' ... SKIPPED. Save method is deactivated')

                    logging.info(' -----> Time step "' + domain_map_time.strftime(time_string_format) + '" ... DONE')

                    logging.info(' ----> Domain "' + domain_name_step + '" ... DONE')

            else:
                logging.info(' ----> Domain "' + domain_name_step + '" ... SKIPPED. Datasets are empty')

        logging.info(' ---> Dump scenario maps [' + time_run.strftime(time_string_format) + '] ... DONE')

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to compute scenario map
    def compute_scenario_map(self, scenario_data_collection):

        time = self.time_run
        geo_data_collection = self.geo_data_collection

        logging.info(' ---> Compute scenario maps [' + time.strftime(time_string_format) + '] ... ')

        scenario_map_collection = {}
        for domain_name_step in self.domain_name_list:

            logging.info(' ----> Domain "' + domain_name_step + '" ... ')

            domain_geo_data = geo_data_collection[domain_name_step]
            domain_scenario_data = scenario_data_collection[domain_name_step]
            domain_section_db = geo_data_collection[domain_name_step][self.domain_sections_db_tag]

            if domain_scenario_data is not None:

                domain_scenario_merged_default = np.zeros(
                    [domain_geo_data[self.domain_scenario_area_tag].shape[0],
                     domain_geo_data[self.domain_scenario_area_tag].shape[1]])
                domain_scenario_merged_default[:, :] = np.nan

                file_path_scenarios_collections = {}
                for section_scenario_id, \
                    (section_scenario_key, section_scenario_data) in enumerate(domain_scenario_data.items()):

                    logging.info(' -----> Section "' + section_scenario_key + '" ... ')

                    section_db_data = None
                    for domain_section_key, domain_section_fields in domain_section_db.items():
                        if domain_section_fields['description'] == section_scenario_key:
                            section_db_data = domain_section_fields.copy()
                            break

                    if section_db_data is not None:

                        section_db_n = section_db_data['n']
                        section_db_description = section_db_data['description']
                        section_db_name_outlet = section_db_data['name_point_outlet']
                        section_db_name_downstream = section_db_data['name_point_downstream']
                        section_db_name_upstream = section_db_data['name_point_upstream']
                        section_db_name_obs = section_db_data['name_point_obs']
                        section_db_idx = section_db_data['idx']

                        assert section_db_description == section_scenario_key

                        if section_scenario_data is not None:

                            section_scenario_trs_cmp = section_scenario_data[self.domain_scenario_index_tag]
                            if self.scenario_analysis is None:
                                section_scenario_times = [time]
                            elif self.scenario_analysis == 'ALL':
                                section_scenario_times = section_scenario_data[self.domain_scenario_time_tag]
                            else:
                                logging.error(' ===> Scenario frequency value "' + str(self.scenario_analysis) +
                                              '" is not allowed')
                                logging.info(' -----> Section "' + section_scenario_key + '" ... FAILED')
                                raise NotImplementedError('Case not implemented yet')

                            for section_scenario_tr_cmp, section_scenario_time in zip(
                                    section_scenario_trs_cmp, section_scenario_times):

                                logging.info(' ------> Time step "' + section_scenario_time.strftime(time_string_format) +
                                             '" ... ')

                                file_path_scenario_ancillary = self.define_file_scenario(
                                    time, self.folder_name_scenario_ancillary,
                                    self.file_name_scenario_ancillary, domain_name_step,
                                    time_step=section_scenario_time)

                                if self.scenario_analysis is None:
                                    if section_scenario_tr_cmp not in list(file_path_scenarios_collections.keys()):
                                        if self.flag_cleaning_ancillary:
                                            if os.path.exists(file_path_scenario_ancillary):
                                                os.remove(file_path_scenario_ancillary)
                                elif (section_scenario_id == 0) and (self.scenario_analysis == 'ALL'):
                                    if section_scenario_tr_cmp not in list(file_path_scenarios_collections.keys()):
                                        if self.flag_cleaning_ancillary:
                                            if os.path.exists(file_path_scenario_ancillary):
                                                os.remove(file_path_scenario_ancillary)

                                # Check tr value
                                if np.isnan(section_scenario_tr_cmp):
                                    section_scenario_tr_other = get_dict_value(domain_scenario_data,
                                                                               self.domain_scenario_index_tag, [])

                                    section_scenario_tr_check = int(np.nanmax(section_scenario_tr_other))
                                else:
                                    section_scenario_tr_check = section_scenario_tr_cmp

                                if section_scenario_tr_check >= self.tr_min:

                                    section_area_idx = np.argwhere(domain_geo_data[self.domain_scenario_area_tag] == section_db_n)

                                    section_scenario_tr_select = max(1, min(self.tr_max, section_scenario_tr_check))

                                    file_path_hazard = self.define_file_hazard(self.folder_name_hazard, self.file_name_hazard,
                                                                               domain_name_step, section_scenario_tr_select)

                                    file_data_hazard = read_file_hazard(
                                        file_path_hazard, file_vars=[self.domain_scenario_hazard_tag])
                                    file_data_h = file_data_hazard[self.domain_scenario_hazard_tag]

                                    idx_x = section_area_idx[:, 0]
                                    idx_y = section_area_idx[:, 1]

                                    if not os.path.exists(file_path_scenario_ancillary):

                                        domain_scenario_merged_filled = deepcopy(domain_scenario_merged_default)

                                        file_data_h_raw = file_data_h[idx_x, idx_y]
                                        file_data_h_scaled = file_data_h_raw / self.scale_factor_hazard

                                        domain_scenario_merged_filled[idx_x, idx_y] = file_data_h_scaled
                                        domain_scenario_merged_filled[domain_scenario_merged_filled <= 0] = np.nan

                                        folder_name, file_name = os.path.split(file_path_scenario_ancillary)
                                        make_folder(folder_name)

                                        write_obj(file_path_scenario_ancillary, domain_scenario_merged_filled)

                                    else:

                                        domain_scenario_merged_tmp = read_obj(file_path_scenario_ancillary)

                                        file_data_h_raw = file_data_h[idx_x, idx_y]
                                        file_data_h_scaled = file_data_h_raw / self.scale_factor_hazard

                                        domain_scenario_merged_tmp[idx_x, idx_y] = file_data_h_scaled
                                        domain_scenario_merged_tmp[domain_scenario_merged_tmp <= 0] = np.nan

                                        if os.path.exists(file_path_scenario_ancillary):
                                            os.remove(file_path_scenario_ancillary)

                                        write_obj(file_path_scenario_ancillary, domain_scenario_merged_tmp)

                                    if section_scenario_time not in list(file_path_scenarios_collections.keys()):
                                        file_path_scenarios_collections[section_scenario_time] = file_path_scenario_ancillary

                                    logging.info(' ------> Time step "' +
                                                 section_scenario_time.strftime(time_string_format) + '" ... DONE')

                                else:
                                    logging.info(' ------> Time step "' +
                                                 section_scenario_time.strftime(time_string_format) +
                                                 '" ... SKIPPED. Scenarios threshold is less then minimum threshold')

                            logging.info(' -----> Section "' + section_scenario_key + '" ... DONE')

                        else:
                            logging.info(' -----> Section "' + section_scenario_key +
                                         '" ... SKIPPED. Section datasets are empty')

                    else:
                        logging.info(' -----> Section "' + section_scenario_key +
                                     '" ... SKIPPED. Section info are empty')

                logging.info(' ----> Domain "' + domain_name_step + '" ... DONE')

                scenario_map_collection[domain_name_step] = file_path_scenarios_collections

            else:
                logging.info(' ----> Domain "' + domain_name_step + '" ... SKIPPED. Domain datasets are empty')
                scenario_map_collection[domain_name_step] = None

        logging.info(' ---> Compute scenario maps [' + time.strftime(time_string_format) + '] ... DONE')

        return scenario_map_collection
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to organize scenario datasets
    def organize_scenario_datasets(self):

        time = self.time_run
        discharge_data_collection = self.discharge_data_collection
        geo_data_collection = self.geo_data_collection

        logging.info(' ---> Organize scenario datasets [' + time.strftime(time_string_format) + '] ... ')

        scenario_info_collection = {}
        scenario_time_collection = {}
        for domain_name_step in self.domain_name_list:

            logging.info(' ----> Domain "' + domain_name_step + '" ... ')

            domain_discharge_data = discharge_data_collection[domain_name_step]
            domain_geo_data = geo_data_collection[domain_name_step]

            domain_discharge_index = geo_data_collection[domain_name_step][self.domain_discharge_index_tag]
            domain_grid_rows = geo_data_collection[domain_name_step][self.domain_grid_x_tag].shape[0]
            domain_grid_cols = geo_data_collection[domain_name_step][self.domain_grid_y_tag].shape[1]
            domain_section_db = geo_data_collection[domain_name_step][self.domain_sections_db_tag]
            
            domain_scenario_workspace = {}
            for section_discharge_key, section_discharge_data in domain_discharge_data.items():

                logging.info(' -----> Section "' + section_discharge_key + '" ... ')

                section_db_data = None
                for domain_section_key, domain_section_fields in domain_section_db.items():
                    if domain_section_fields['description'] == section_discharge_key:
                        section_db_data = domain_section_fields.copy()
                        break

                if (section_db_data is not None) and (section_discharge_data is not None):

                    section_db_n = section_db_data['n']
                    section_db_description = section_db_data['description']
                    section_db_name_outlet = section_db_data['name_point_outlet']
                    section_db_name_obs = section_db_data['name_point_obs']
                    section_db_idx = section_db_data['idx']

                    assert section_db_description == section_discharge_key

                    # Compute scenario idx
                    section_discharge_idx = domain_discharge_index[section_db_idx[0] - 1, section_db_idx[1] - 1]

                    if self.scenario_analysis is None:

                        # Compute discharge for evaluating scenario
                        section_discharge_run, section_discharge_time, \
                            section_discharge_value, section_n_value = self.compute_scenario_discharge(section_discharge_data)
                        # Compute tr for evaluating scenario
                        section_scenario_tr = self.compute_scenario_tr(section_discharge_idx, section_discharge_value)

                        domain_scenario_workspace[section_discharge_key] = {}
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_index_tag] = section_scenario_tr
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_discharge_tag] = section_discharge_value
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_time_tag] = section_discharge_time
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_n_tag] = section_n_value

                        logging.info(' -----> Section "' + section_discharge_key + '" ... DONE')

                    elif self.scenario_analysis == 'ALL':

                        # Compute discharge for evaluating scenario
                        section_discharge_runs, section_discharge_times, \
                            section_discharge_values, section_n_values = self.compute_scenario_discharge(
                                section_discharge_data,
                                analysis_freq=self.scenario_analysis, analysis_method=self.scenario_method)

                        # Compute tr for evaluating scenario
                        section_scenario_trs = self.compute_scenario_tr(section_discharge_idx, section_discharge_values)

                        domain_scenario_workspace[section_discharge_key] = {}
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_index_tag] = section_scenario_trs
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_discharge_tag] = section_discharge_values
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_time_tag] = section_discharge_times
                        domain_scenario_workspace[section_discharge_key][self.domain_scenario_n_tag] = section_n_values

                        logging.info(' -----> Section "' + section_discharge_key + '" ... DONE')

                    else:

                        logging.error(' ===> Scenario frequency value "' + str(self.scenario_analysis) +
                                      '" is not allowed')
                        logging.info(' -----> Section "' + section_discharge_key + '" ... FAILED')
                        raise NotImplementedError('Case not implemented yet')

                else:

                    logging.info(' -----> Section "' + section_discharge_key + '" ... SKIPPED. Datasets are empty')
                    domain_scenario_workspace[section_discharge_key] = None

            scenario_info_collection[domain_name_step] = domain_scenario_workspace

            logging.info(' ----> Domain "' + domain_name_step + '" ... DONE')

        logging.info(' ---> Organize scenario datasets [' + time.strftime(time_string_format) + '] ... DONE')

        return scenario_info_collection

    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
