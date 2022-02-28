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

from lib_utils_hazard import read_file_hazard
from lib_utils_io import read_obj, write_obj, save_file_json
from lib_utils_system import fill_tags2string, make_folder
from lib_utils_generic import get_dict_value, reduce_dict_2_lists
from lib_utils_plot import save_file_tiff, save_file_png, read_file_tiff, save_file_info
from lib_utils_ts import prepare_file_ts, save_file_ts
from lib_utils_tr import cmp_tr_exp, cmp_tr_linear

from lib_info_args import logger_name, time_format_algorithm

# Logging
log_stream = logging.getLogger(logger_name)
# Debug
import matplotlib.pylab as plt
######################################################################################


# -------------------------------------------------------------------------------------
# Class DriverScenario
class DriverScenario:

    # -------------------------------------------------------------------------------------
    # Initialize class
    def __init__(self, time_now, time_run, discharge_data_collection, geo_data_collection,
                 src_dict, anc_dict, dst_dict,
                 alg_ancillary=None, alg_template_tags=None,
                 flag_telemac_data='telemac_data', flag_hazard_data='hazard_data',
                 flag_scenario_data_info='scenario_data_info',
                 flag_scenario_data_file='scenario_data_file',
                 flag_scenario_data_map='scenario_data_map',
                 flag_scenario_plot_info='scenario_info',
                 flag_scenario_plot_tiff='scenario_tiff',
                 flag_scenario_plot_png='scenario_png',
                 flag_scenario_data_ts='scenario_time_series',
                 flag_cleaning_anc_scenario_info=True, flag_cleaning_anc_scenario_file=True,
                 flag_cleaning_anc_scenario_map=True,
                 flag_cleaning_plot_scenario=True, flag_cleaning_data_scenario=True):

        self.time_now = time_now
        self.time_run = time_run

        self.discharge_data_collection = discharge_data_collection
        self.geo_data_collection = geo_data_collection

        self.flag_telemac_data = flag_telemac_data
        self.flag_hazard_data = flag_hazard_data
        self.flag_scenario_data_info = flag_scenario_data_info
        self.flag_scenario_data_file = flag_scenario_data_file
        self.flag_scenario_data_map = flag_scenario_data_map
        self.flag_scenario_plot_info = flag_scenario_plot_info
        self.flag_scenario_plot_tiff = flag_scenario_plot_tiff
        self.flag_scenario_plot_png = flag_scenario_plot_png
        self.flag_scenario_data_ts = flag_scenario_data_ts

        self.alg_ancillary = alg_ancillary
        self.tr_min = alg_ancillary['tr_min']
        self.tr_max = alg_ancillary['tr_max']
        self.tr_freq = alg_ancillary['tr_freq']

        self.scenario_analysis = alg_ancillary['scenario_analysis']
        self.scenario_type = alg_ancillary['scenario_type']
        if 'scenario_tiling' in list(alg_ancillary.keys()):
            self.scenario_tiling = alg_ancillary['scenario_tiling']
        else:
            self.scenario_tiling = 'rounded'

        self.alg_template_tags = alg_template_tags
        self.file_name_tag = 'file_name'
        self.folder_name_tag = 'folder_name'
        self.save_status_tag = 'save_status'
        self.group_dst_plot = 'plot'
        self.group_dst_data = 'data'

        self.domain_name_list = self.alg_ancillary['domain_name']

        self.folder_name_telemac = src_dict[self.flag_telemac_data][self.folder_name_tag]
        self.file_name_telemac = src_dict[self.flag_telemac_data][self.file_name_tag]

        self.folder_name_hazard = src_dict[self.flag_hazard_data][self.folder_name_tag]
        self.file_name_hazard = src_dict[self.flag_hazard_data][self.file_name_tag]

        self.folder_name_scenario_anc_info = anc_dict[self.flag_scenario_data_info][self.folder_name_tag]
        self.file_name_scenario_anc_info = anc_dict[self.flag_scenario_data_info][self.file_name_tag]
        self.folder_name_scenario_anc_file = anc_dict[self.flag_scenario_data_file][self.folder_name_tag]
        self.file_name_scenario_anc_file = anc_dict[self.flag_scenario_data_file][self.file_name_tag]
        self.folder_name_scenario_anc_map = anc_dict[self.flag_scenario_data_map][self.folder_name_tag]
        self.file_name_scenario_anc_map = anc_dict[self.flag_scenario_data_map][self.file_name_tag]

        self.file_path_scenario_anc_info = self.define_file_scenario(
            self.time_now, self.folder_name_scenario_anc_info, self.file_name_scenario_anc_info,
            file_type='dictionary')

        self.file_path_scenario_anc_file = self.define_file_scenario(
            self.time_now, self.folder_name_scenario_anc_file, self.file_name_scenario_anc_file,
            file_type='dictionary')

        self.format_tr = '{:03d}'
        self.scenario_tr = self.define_tr_scenario(self.tr_min, self.tr_max, self.tr_freq)

        group_dst_plot = dst_dict[self.group_dst_plot]
        self.folder_name_scenario_plot_info = group_dst_plot[self.flag_scenario_plot_info][self.folder_name_tag]
        self.file_name_scenario_plot_info = group_dst_plot[self.flag_scenario_plot_info][self.file_name_tag]
        self.save_status_scenario_plot_info = group_dst_plot[self.flag_scenario_plot_info][self.save_status_tag]
        self.folder_name_scenario_plot_tiff = group_dst_plot[self.flag_scenario_plot_tiff][self.folder_name_tag]
        self.file_name_scenario_plot_tiff = group_dst_plot[self.flag_scenario_plot_tiff][self.file_name_tag]
        self.save_status_scenario_plot_tiff = group_dst_plot[self.flag_scenario_plot_tiff][self.save_status_tag]
        self.folder_name_scenario_plot_png = group_dst_plot[self.flag_scenario_plot_png][self.folder_name_tag]
        self.file_name_scenario_plot_png = group_dst_plot[self.flag_scenario_plot_png][self.file_name_tag]
        self.save_status_scenario_plot_png = group_dst_plot[self.flag_scenario_plot_png][self.save_status_tag]

        group_dst_data = dst_dict[self.group_dst_data]
        self.folder_name_scenario_data_ts = group_dst_data[self.flag_scenario_data_ts][self.folder_name_tag]
        self.file_name_scenario_data_ts = group_dst_data[self.flag_scenario_data_ts][self.file_name_tag]
        self.save_status_scenario_data_ts = group_dst_data[self.flag_scenario_data_ts][self.save_status_tag]

        self.flag_cleaning_anc_scenario_info = flag_cleaning_anc_scenario_info
        self.flag_cleaning_anc_scenario_file = flag_cleaning_anc_scenario_file
        self.flag_cleaning_anc_scenario_map = flag_cleaning_anc_scenario_map
        self.flag_cleaning_plot_scenario = flag_cleaning_plot_scenario
        self.flag_cleaning_data_scenario = flag_cleaning_data_scenario

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
        self.domain_scenario_index_right_tag = 'scenario_idx_right'
        self.domain_scenario_index_left_tag = 'scenario_idx_left'
        self.domain_scenario_weight_right_tag = 'scenario_weight_right'
        self.domain_scenario_weight_left_tag = 'scenario_weight_left'
        self.domain_scenario_discharge_tag = 'discharge_value'
        self.domain_scenario_type_tag = 'type_value'
        self.domain_scenario_time_tag = 'time'
        self.domain_scenario_n_tag = 'scenario_n'
        self.domain_scenario_attrs_tag = 'scenario_attrs'

        self.domain_scenario_index_cmp_tag = 'scenario_idx_cmp'
        self.domain_scenario_discharge_cmp_tag = 'discharge_value_cmp'
        self.domain_scenario_h_cmp_tag = 'h_value_cmp'

        self.domain_scenario_area_tag = "mappa_aree_new"
        self.domain_scenario_grid_x_tag = "new_x"
        self.domain_scenario_grid_y_tag = "new_y"

        self.domain_scenario_hazard_name = 'mappa_h'
        self.domain_scenario_hazard_format = np.float32
        self.domain_scenario_hazard_scale_factor = 1
        self.domain_scenario_hazard_units = 'm'

        self.domain_name_tag = 'domain_name'

        self.var_name_time = 'time'
        self.var_name_discharge = 'discharge'
        self.var_name_water_level = 'water_level'
        self.var_name_type = 'type'

        self.dframe_columns = [self.domain_scenario_index_cmp_tag, self.domain_scenario_discharge_cmp_tag,
                               self.domain_scenario_h_cmp_tag]

        self.correction_discharge_factor = 1.16
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
    def compute_scenario_tr(self, section_discharge_idx, section_discharge_times, section_discharge_values,
                            section_discharge_name, section_scenario_tr_min=1, section_scenario_tr_max=500):

        if not isinstance(section_discharge_values, list):
            section_discharge_values = [section_discharge_values]
        if not isinstance(section_discharge_times, list):
            section_discharge_times = [section_discharge_times]

        if section_discharge_idx > 0.0:

            section_scenario_trs = []
            section_scenario_trs_right, section_scenario_trs_left = [], []
            section_scenario_weights_right, section_scenario_weights_left = [], []
            section_scenario_tr_check, section_scenario_tr_right_check, section_scenario_tr_left_check = [], [], []
            for section_discharge_id, (section_discharge_time, section_discharge_value) in enumerate(
                    zip(section_discharge_times, section_discharge_values)):

                if section_discharge_value >= 0.0:

                    if section_discharge_value >= section_discharge_idx * self.correction_discharge_factor:
                        section_scenario_tr_rounded, \
                            section_scenario_tr_right, section_scenario_tr_left, \
                            section_scenario_weight_right, section_scenario_weight_left = cmp_tr_exp(
                                section_discharge_idx, section_discharge_value)
                    else:

                        section_scenario_tr_rounded, \
                            section_scenario_tr_right, section_scenario_tr_left, \
                            section_scenario_weight_right, section_scenario_weight_left = cmp_tr_linear(
                                section_discharge_idx, section_discharge_value,
                                section_discharge_factor=self.correction_discharge_factor)

                    section_scenario_tr = int(section_scenario_tr_rounded)
                    section_scenario_tr_right = int(section_scenario_tr_right)
                    section_scenario_tr_left = int(section_scenario_tr_left)

                    if section_scenario_tr < section_scenario_tr_min:
                        section_scenario_tr = section_scenario_tr_min
                        section_scenario_tr_check.append(section_discharge_time.strftime(time_format_algorithm))
                    elif section_scenario_tr > section_scenario_tr_max:
                        log_stream.error(' ===> At time "' + section_discharge_time.strftime(time_format_algorithm) +
                                         '" find the "tr ' + str(section_scenario_tr) +
                                         '" greater then "tr_max ' + str(section_scenario_tr_max) + '"')
                        raise NotImplementedError('Case not implemented yet')

                    if section_scenario_tr_right < section_scenario_tr_min:
                        section_scenario_tr_right_check.append(section_discharge_time.strftime(time_format_algorithm))
                        section_scenario_tr_right = section_scenario_tr_min
                        section_scenario_tr_left = section_scenario_tr_min
                        section_scenario_weight_right = 0.5
                        section_scenario_weight_left = 0.5

                    elif section_scenario_tr_left > section_scenario_tr_max:
                        log_stream.error(' ===> At time "' + section_discharge_time.strftime(time_format_algorithm) +
                                         '" find the "tr_left ' + str(section_scenario_tr_left) +
                                         '" greater then "tr_max ' + str(section_scenario_tr_max) + '"')
                        raise NotImplementedError('Case not implemented yet')

                else:
                    section_scenario_tr = np.nan
                    section_scenario_tr_right, section_scenario_tr_left = np.nan, np.nan
                    section_scenario_weight_right, section_scenario_weight_left = np.nan, np.nan

                section_scenario_trs.append(section_scenario_tr)
                section_scenario_trs_right.append(section_scenario_tr_right)
                section_scenario_trs_left.append(section_scenario_tr_left)
                section_scenario_weights_right.append(section_scenario_weight_right)
                section_scenario_weights_left.append(section_scenario_weight_left)

            if section_scenario_tr_check:
                section_scenario_tr_str = ', '.join(section_scenario_tr_check)
                log_stream.warning(' ===> At times "' + section_scenario_tr_str +
                                   '" found the "tr" less then "tr_min ' + str(section_scenario_tr_min) + '"')
                log_stream.warning(' ===> Set the "tr" equal to "tr_min"')
            if section_scenario_tr_right_check:
                section_scenario_tr_str = ', '.join(section_scenario_tr_right_check)
                log_stream.warning(' ===> At times "' + section_scenario_tr_str +
                                   '" found the "tr_right" less then "tr_min ' + str(section_scenario_tr_min) + '"')
                log_stream.warning(' ===> Set the "tr_right" equal to "tr_min"')
        else:
            section_scenario_trs = [np.nan] * section_discharge_values.__len__()
            section_scenario_trs_right = [np.nan] * section_discharge_values.__len__()
            section_scenario_trs_left = [np.nan] * section_discharge_values.__len__()
            section_scenario_weights_right = [np.nan] * section_discharge_values.__len__()
            section_scenario_weights_left = [np.nan] * section_discharge_values.__len__()

        return section_scenario_trs, section_scenario_trs_right, section_scenario_trs_left, \
            section_scenario_weights_right, section_scenario_weights_left

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to compute discharge for evaluating scenario
    @staticmethod
    def compute_scenario_discharge(dframe_discharge, dframe_type, analysis_freq=None):

        if analysis_freq == 'max_period':

            reference_value, time_value, occurrence_value, discharge_value, type_value = [], [], [], [], []

            time_max_value = dframe_discharge.idxmax()
            time_max_idx = dframe_discharge.index.get_loc(time_max_value)

            reference_value.append(time_max_idx)
            time_value.append(time_max_value)
            discharge_value.append(dframe_discharge[time_max_idx])
            type_value.append(dframe_type[time_max_idx])
            occurrence_value.append(1)

        elif analysis_freq == 'all_period':

            reference_value, time_value, occurrence_value, discharge_value, type_value = [], [], [], [], []
            for id_step, (time_step, discharge_step, type_step) in enumerate(zip(
                    dframe_discharge.index, dframe_discharge.values, dframe_type.values)):

                if not np.isnan(discharge_step):
                    reference_value.append(id_step)
                    time_value.append(time_step)
                    discharge_value.append(discharge_step)
                    type_value.append(type_step)
                    occurrence_value.append(1)
        else:
            log_stream.error(' ===> Frequency to compute discharge for evaluating scenario is not defined')
            raise NotImplemented('Method not implemented yet')

        return reference_value, time_value, discharge_value, type_value, occurrence_value
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to define hazard file
    def define_file_scenario(self, time_run, folder_name_raw, file_name_raw, domain_list=None,
                             file_type='string', time_step=None):

        template_tags = self.alg_template_tags

        if time_step is None:
            time_step = time_run

        if domain_list is None:
            domain_list = self.domain_name_list
        if not isinstance(domain_list, list):
            domain_list = [domain_list]

        file_path_dict = {}
        for domain_name in domain_list:
            template_values_step = {'domain_name': domain_name,
                                    'destination_sub_path_time_scenario_plot': time_run,
                                    'destination_sub_path_time_scenario_data': time_run,
                                    'ancillary_sub_path_time_scenario': time_run,
                                    'destination_datetime_scenario_plot': time_step,
                                    'destination_datetime_scenario_data': time_step,
                                    'ancillary_datetime_scenario': time_step
                                    }

            folder_name_def = fill_tags2string(folder_name_raw, template_tags, template_values_step)
            file_name_def = fill_tags2string(file_name_raw, template_tags, template_values_step)
            path_name_def = os.path.join(folder_name_def, file_name_def)

            file_path_dict[domain_name] = path_name_def

        if file_type == 'string':
            _, file_path_list = reduce_dict_2_lists(file_path_dict)

            if file_path_list.__len__() == 1:
                file_path_obj = file_path_list[0]
            else:
                log_stream.error(' ===> File format is not supported')
                raise NotImplementedError('File multi-band not implemented yet')

        elif file_type == 'dictionary':
            file_path_obj = deepcopy(file_path_dict)
        else:
            log_stream.error(' ===> File type mode is not supported')
            raise NotImplementedError('File multi-band not implemented yet')

        return file_path_obj

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to dump scenario map
    def dump_scenario_map(self, scenario_map_collection, scenario_info_collection, scenario_dframe_collection):

        time_run = self.time_run
        time_now = self.time_now

        time_now_string = time_now.strftime(time_format_algorithm)
        geo_data_collection = self.geo_data_collection
        scenario_description_collection = self.scenario_description_collection

        log_stream.info(' ---> Dump scenario time-series, maps and info [' + time_run.strftime(time_format_algorithm) +
                        '] ... ')

        for domain_name_step in self.domain_name_list:

            log_stream.info(' ----> Domain "' + domain_name_step + '" ... ')

            domain_geo_collection = geo_data_collection[domain_name_step]
            domain_info_collection = scenario_info_collection[domain_name_step]
            domain_map_collection = scenario_map_collection[domain_name_step]
            domain_dframe_collection = scenario_dframe_collection[domain_name_step]
            domain_description_collection = scenario_description_collection[domain_name_step]

            if not domain_map_collection:
                domain_map_collection = None

            if domain_map_collection is not None:

                log_stream.info(' -----> Time-Series section  ... ')
                log_stream.info(' ------> Prepare file data  ... ')
                file_path_scenario_data_ts = self.define_file_scenario(
                    time_now, self.folder_name_scenario_data_ts, self.file_name_scenario_data_ts,
                    domain_name_step, file_type='string', time_step=time_now)

                if self.flag_cleaning_data_scenario:
                    if os.path.exists(file_path_scenario_data_ts):
                        os.remove(file_path_scenario_data_ts)

                domain_ts_data = prepare_file_ts(domain_description_collection, domain_dframe_collection,
                                                 domain_info_collection)

                log_stream.info(' ------> Prepare file data  ... DONE')

                # Save information in ts file
                folder_name_scenario_data_ts, file_name_scenario_data_ts = os.path.split(
                    file_path_scenario_data_ts)
                make_folder(folder_name_scenario_data_ts)

                log_stream.info(' ------> Save file ts ' + file_name_scenario_data_ts + ' ... ')
                if self.save_status_scenario_data_ts:
                    if not os.path.exists(file_path_scenario_data_ts):
                        save_file_ts(file_path_scenario_data_ts, domain_ts_data)
                        log_stream.info(' -------> Save file ts ' + file_name_scenario_data_ts + ' ... DONE')
                    else:
                        log_stream.info(' ------> Save file ts ' + file_name_scenario_data_ts + ' ... PREVIOUSLY SAVED')
                else:
                    log_stream.info(' ------> Save file ts ' + file_name_scenario_data_ts +
                                    ' ... SKIPPED. Save method is deactivated')

                log_stream.info(' ------> Save file ts ' + file_name_scenario_data_ts + ' ... DONE')
                log_stream.info(' -----> Time-Series section  ... DONE')

                log_stream.info(' -----> Maps section  ... ')
                for domain_map_time, domain_map_file_ancillary in sorted(domain_map_collection.items()):

                    log_stream.info(' ------> Time step "' + domain_map_time.strftime(time_format_algorithm) + '" ... ')

                    log_stream.info(' -------> Prepare file data ... ')

                    if domain_map_file_ancillary.endswith('tiff') or \
                            domain_map_file_ancillary.endswith('tif'):

                        domain_map_data = read_file_tiff(domain_map_file_ancillary)

                        # DEBUG START
                        # plt.figure()
                        # plt.imshow(domain_map_data)
                        # plt.colorbar()
                        # plt.clim(0, 8)
                        # plt.show()
                        # DEBUG END

                    elif domain_map_file_ancillary.endswith('workspace'):
                        domain_map_data = read_obj(domain_map_file_ancillary)
                    else:
                        log_stream.error(' ===> Read selected method is not supported.')
                        raise NotImplementedError('Case not implemented yet')

                    file_path_scenario_plot_info = self.define_file_scenario(
                        time_now, self.folder_name_scenario_plot_info, self.file_name_scenario_plot_info,
                        domain_name_step, file_type='string', time_step=domain_map_time)
                    file_path_scenario_plot_tiff = self.define_file_scenario(
                        time_now, self.folder_name_scenario_plot_tiff, self.file_name_scenario_plot_tiff,
                        domain_name_step, file_type='string', time_step=domain_map_time)
                    file_path_scenario_plot_png = self.define_file_scenario(
                        time_now, self.folder_name_scenario_plot_png, self.file_name_scenario_plot_png,
                        domain_name_step, file_type='string', time_step=domain_map_time)

                    if self.flag_cleaning_plot_scenario:
                        if os.path.exists(file_path_scenario_plot_info):
                            os.remove(file_path_scenario_plot_info)
                        if os.path.exists(file_path_scenario_plot_tiff):
                            os.remove(file_path_scenario_plot_tiff)
                        if os.path.exists(file_path_scenario_plot_png):
                            os.remove(file_path_scenario_plot_png)

                    domain_geo_data = domain_geo_collection[self.domain_scenario_area_tag]
                    domain_geo_x = domain_geo_collection[self.domain_scenario_grid_x_tag]
                    domain_geo_y = domain_geo_collection[self.domain_scenario_grid_y_tag]

                    time_step_string = domain_map_time.strftime(time_format_algorithm)

                    section_info_collection = {}
                    for domain_info_key, domain_info_fields in domain_info_collection.items():
                        if domain_info_fields is not None:
                            if domain_info_key in domain_description_collection:

                                if self.domain_scenario_attrs_tag in list(domain_info_fields.keys()):
                                    section_info_attrs = domain_info_fields[self.domain_scenario_attrs_tag]
                                else:
                                    log_stream.warning(' ===> Section attributes for "' + domain_info_key +
                                                       '" are undefined due to time-series discharge datasets.')
                                    section_info_attrs = {}

                                domain_info_dframe = pd.DataFrame(domain_info_fields, index=domain_info_fields['time'])

                                if not domain_info_dframe[domain_info_dframe.index.isin([domain_map_time])].empty:
                                    section_info_fields = domain_info_dframe[domain_info_dframe.index.isin([domain_map_time])].to_dict('r')[0]
                                else:
                                    section_info_fields = {}
                                    log_stream.warning(' ===> Section information for "' + domain_info_key +
                                                       '" are undefined due to time-series discharge datasets.')

                                if isinstance(section_info_fields, dict) and isinstance(section_info_attrs, dict):
                                    section_info_fields = {**section_info_fields, **section_info_attrs}
                                else:
                                    log_stream.warning(' ===> Section information and attributes for "' + domain_info_key +
                                                       '" are undefined due to time-series discharge datasets.')
                                    section_info_fields = {}

                                if section_info_fields is not None:
                                    for section_info_key, section_info_value in section_info_fields.items():
                                        if isinstance(section_info_value, pd.Timestamp):
                                            section_tmp_value = section_info_value.strftime(time_format_algorithm)
                                            section_info_fields[section_info_key] = section_tmp_value
                                        elif isinstance(section_info_value, list):
                                           section_tmp_value = ','.join(str(elem) for elem in section_info_value)
                                           section_info_fields[section_info_key] = section_tmp_value
                                        elif isinstance(section_info_value, bool):
                                            section_tmp_value = str(section_info_value)
                                            section_info_fields[section_info_key] = section_tmp_value
                                section_info_collection[domain_info_key] = section_info_fields

                    section_info_collection['scenario_name'] = domain_name_step
                    section_info_collection['scenario_time_now'] = time_now_string
                    section_info_collection['scenario_time_step'] = time_step_string

                    log_stream.info(' -------> Prepare file data ... DONE')

                    # Save information in json file
                    folder_name_scenario_plot_info, file_name_scenario_plot_info = os.path.split(
                        file_path_scenario_plot_info)
                    make_folder(folder_name_scenario_plot_info)

                    log_stream.info(' -------> Save file json ' + file_name_scenario_plot_info + ' ... ')
                    if self.save_status_scenario_plot_info:
                        if not os.path.exists(file_path_scenario_plot_info):
                            save_file_info(file_path_scenario_plot_info, section_info_collection)
                            log_stream.info(' -------> Save file json ' + file_name_scenario_plot_info +
                                            ' ... DONE')
                        else:
                            log_stream.info(' -------> Save file json ' + file_name_scenario_plot_info +
                                            ' ... PREVIOUSLY SAVED')
                    else:
                        log_stream.info(' -------> Save file json ' + file_name_scenario_plot_info +
                                        ' ... SKIPPED. Save method is deactivated')

                    # Save information in png file
                    folder_name_scenario_plot_png, file_name_scenario_plot_png = os.path.split(
                        file_path_scenario_plot_png)
                    make_folder(folder_name_scenario_plot_png)

                    log_stream.info(' -------> Save file png ' + file_name_scenario_plot_png + ' ... ')
                    if self.save_status_scenario_plot_png:
                        if not os.path.exists(file_path_scenario_plot_png):
                            save_file_png(file_path_scenario_plot_png,
                                          domain_map_data, domain_geo_x, domain_geo_y,
                                          scenario_name=domain_name_step,
                                          scenario_time_now_string=time_now_string,
                                          scenario_time_step_string=time_step_string,
                                          fig_color_map_type=None, fig_dpi=150)
                            log_stream.info(' -------> Save file png ' + file_name_scenario_plot_png +
                                            ' ... DONE')
                        else:
                            log_stream.info(' -------> Save file png ' + file_name_scenario_plot_png +
                                            ' ... PREVIOUSLY SAVED')
                    else:
                        log_stream.info(' -------> Save file png ' + file_name_scenario_plot_png +
                                        ' ... SKIPPED. Save method is deactivated')

                    # Save information in tiff file
                    folder_name_scenario_plot_tiff, file_name_scenario_plot_tiff = os.path.split(
                        file_path_scenario_plot_tiff)
                    make_folder(folder_name_scenario_plot_tiff)

                    log_stream.info(' -------> Save file tiff ' + file_name_scenario_plot_tiff + ' ... ')
                    if self.save_status_scenario_plot_tiff:
                        if not os.path.exists(file_path_scenario_plot_tiff):
                            save_file_tiff(file_path_scenario_plot_tiff,
                                           domain_map_data, domain_geo_x, domain_geo_y,
                                           file_epsg_code='EPSG:32632')
                            log_stream.info(' -------> Save file tiff ' + file_name_scenario_plot_tiff +
                                            ' ... DONE')
                        else:
                            log_stream.info(' -------> Save file tiff ' + file_name_scenario_plot_tiff +
                                            ' ... PREVIOUSLY SAVED')
                    else:
                        log_stream.info(' -------> Save file tiff ' + file_name_scenario_plot_tiff +
                                        ' ... SKIPPED. Save method is deactivated')

                    log_stream.info(' ------> Time step "' + domain_map_time.strftime(time_format_algorithm) +
                                    '" ... DONE')

                log_stream.info(' -----> Maps section  ... DONE')
                log_stream.info(' ----> Domain "' + domain_name_step + '" ... DONE')

            else:
                log_stream.info(' ----> Domain "' + domain_name_step + '" ... SKIPPED. Datasets are empty')

        log_stream.info(' ---> Dump scenario time-series, maps and info [' + time_run.strftime(time_format_algorithm) +
                        '] ... DONE')

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to compute scenario map
    def compute_scenario_map(self, scenario_data_collection):

        time = self.time_run
        geo_data_collection = self.geo_data_collection

        file_path_scenario_anc_collections_file = self.file_path_scenario_anc_file

        log_stream.info(' ---> Compute scenario maps [' + time.strftime(time_format_algorithm) + '] ... ')

        scenario_map_collection = {}
        scenario_dframe_collection = {}
        for domain_name_step in self.domain_name_list:

            log_stream.info(' ----> Domain "' + domain_name_step + '" ... ')

            domain_geo_collection = geo_data_collection[domain_name_step]
            domain_scenario_data = scenario_data_collection[domain_name_step]
            domain_section_db = geo_data_collection[domain_name_step][self.domain_sections_db_tag]

            file_path_scenario_anc_domain_file = file_path_scenario_anc_collections_file[domain_name_step]

            if domain_scenario_data is not None:

                domain_scenario_merged_default = np.zeros(
                    [domain_geo_collection[self.domain_scenario_area_tag].shape[0],
                     domain_geo_collection[self.domain_scenario_area_tag].shape[1]])
                domain_scenario_merged_default[:, :] = np.nan

                domain_geo_data = domain_geo_collection[self.domain_scenario_area_tag]
                domain_geo_x = domain_geo_collection[self.domain_scenario_grid_x_tag]
                domain_geo_y = domain_geo_collection[self.domain_scenario_grid_y_tag]

                if self.flag_cleaning_anc_scenario_info or self.flag_cleaning_anc_scenario_file or \
                        self.flag_cleaning_anc_scenario_map:
                    file_path_scenario_tmp = []
                    if os.path.exists(file_path_scenario_anc_domain_file):
                        file_path_scenario_obj = read_obj(file_path_scenario_anc_domain_file)
                        file_path_scenario_tmp = list(file_path_scenario_obj['file_path'].values())
                        os.remove(file_path_scenario_anc_domain_file)

                    for file_path_step in file_path_scenario_tmp:
                        if os.path.exists(file_path_step):
                            os.remove(file_path_step)

                if not os.path.exists(file_path_scenario_anc_domain_file):

                    section_dframe_collections = {}
                    file_path_scenarios_collections = {}
                    for section_scenario_id, \
                            (section_scenario_key, section_scenario_data) in enumerate(domain_scenario_data.items()):

                        # DEBUG
                        # section_scenario_key = 'Entella_Carasco'
                        # section_scenario_data = domain_scenario_data[section_scenario_key]

                        log_stream.info(' -----> Section "' + section_scenario_key + '" ... ')

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
                            section_db_idx_terrain = section_db_data['idx_data_terrain']

                            if 'idx_data_hydraulic' in list(section_db_data.keys()):
                                section_db_idx_hydraulic = section_db_data['idx_data_hydraulic']
                            else:
                                section_db_idx_hydraulic = None

                            assert section_db_description == section_scenario_key

                            if section_scenario_data is not None:

                                section_scenario_discharges = section_scenario_data[self.domain_scenario_discharge_tag]
                                section_scenario_trs_cmp = section_scenario_data[self.domain_scenario_index_tag]
                                section_scenario_trs_right = section_scenario_data[self.domain_scenario_index_right_tag]
                                section_scenario_trs_left = section_scenario_data[self.domain_scenario_index_left_tag]
                                section_scenario_wgs_right = section_scenario_data[self.domain_scenario_weight_right_tag]
                                section_scenario_wgs_left = section_scenario_data[self.domain_scenario_weight_left_tag]

                                if self.scenario_analysis == 'max_period':
                                    section_scenario_times = [time]
                                elif self.scenario_analysis == 'all_period':
                                    section_scenario_times = section_scenario_data[self.domain_scenario_time_tag]
                                else:
                                    log_stream.error(' ===> Scenario frequency value "' + str(self.scenario_analysis) +
                                                     '" is not allowed')
                                    log_stream.info(' -----> Section "' + section_scenario_key + '" ... FAILED')
                                    raise NotImplementedError('Case not implemented yet')

                                section_scenario_time_list, section_scenario_discharge_list = [], []
                                section_scenario_tr_list, section_scenario_h_list = [], []
                                section_scenario_tr_list_right, section_scenario_tr_list_left = [], []
                                for id_scenario_time, section_scenario_time in enumerate(section_scenario_times):

                                    section_scenario_discharge = section_scenario_discharges[id_scenario_time]
                                    section_scenario_tr_cmp = section_scenario_trs_cmp[id_scenario_time]
                                    section_scenario_tr_right = section_scenario_trs_right[id_scenario_time]
                                    section_scenario_tr_left = section_scenario_trs_left[id_scenario_time]
                                    section_scenario_wg_right = section_scenario_wgs_right[id_scenario_time]
                                    section_scenario_wg_left = section_scenario_wgs_left[id_scenario_time]

                                    log_stream.info(' ------> Time step "' + section_scenario_time.strftime(
                                        time_format_algorithm) + '" ... ')

                                    file_path_scenario_anc_map = self.define_file_scenario(
                                        time, self.folder_name_scenario_anc_map,
                                        self.file_name_scenario_anc_map,
                                        domain_list=domain_name_step, file_type='string',
                                        time_step=section_scenario_time)

                                    '''
                                    if self.scenario_analysis is None:
                                        if section_scenario_tr_cmp not in list(file_path_scenarios_collections.keys()):
                                            if self.flag_cleaning_ancillary:
                                                if os.path.exists(flag_cleaning_plot_scenario_maps):
                                                    os.remove(file_path_scenario_ancillary)
                                    elif (section_scenario_id == 0) and (self.scenario_analysis == 'ALL'):
                                        if section_scenario_tr_cmp not in list(file_path_scenarios_collections.keys()):
                                            if self.flag_cleaning_ancillary:
                                                if os.path.exists(file_path_scenario_ancillary):
                                                    os.remove(file_path_scenario_ancillary)
                                    '''

                                    # Find tr value
                                    if np.isnan(section_scenario_tr_cmp):

                                        section_scenario_tr_other = get_dict_value(
                                            domain_scenario_data, self.domain_scenario_index_tag, [])
                                        section_scenario_tr_check = int(np.nanmax(section_scenario_tr_other))
                                        section_scenario_tr_right_check = deepcopy(section_scenario_tr_check)
                                        section_scenario_tr_left_check = deepcopy(section_scenario_tr_check)
                                        section_scenario_wg_right_check, section_scenario_wg_left_check = 1.0, 1.0

                                        log_stream.warning(' ===> Scenario tr is undefined for section "' +
                                                           section_db_description + '". Check the datasets')

                                    else:
                                        section_scenario_tr_check = section_scenario_tr_cmp
                                        section_scenario_tr_right_check = section_scenario_tr_right
                                        section_scenario_tr_left_check = section_scenario_tr_left
                                        section_scenario_wg_right_check = section_scenario_wg_right
                                        section_scenario_wg_left_check = section_scenario_wg_left

                                    # Compare tr value with tr min
                                    if section_scenario_tr_check >= self.tr_min:

                                        section_area_idx = np.argwhere(
                                            domain_geo_collection[self.domain_scenario_area_tag] == section_db_n)

                                        section_scenario_tr_select = max(self.tr_min,
                                                                         min(self.tr_max, section_scenario_tr_check))
                                        section_scenario_tr_right_select = max(self.tr_min,
                                                                               min(self.tr_max, section_scenario_tr_right_check))
                                        section_scenario_tr_left_select = max(self.tr_min,
                                                                              min(self.tr_max, section_scenario_tr_left_check))

                                        if section_scenario_tr_right_select > section_scenario_tr_left_select:
                                            log_stream.error(' ===> Left scenario must be greater then right scenario')
                                            raise RuntimeError('Check the evaluation of scenario boundaries')

                                        if self.scenario_tiling == 'rounded':

                                            file_path_hazard = self.define_file_hazard(
                                                self.folder_name_hazard, self.file_name_hazard,
                                                domain_name_step, section_scenario_tr_select)

                                            file_data_hazard = read_file_hazard(
                                                file_path_hazard,
                                                file_vars=[self.domain_scenario_hazard_name],
                                                file_format=[self.domain_scenario_hazard_format],
                                                file_scale_factor=[self.domain_scenario_hazard_scale_factor])

                                            if file_data_hazard is not None:
                                                file_data_h = file_data_hazard[self.domain_scenario_hazard_name]
                                            else:
                                                log_stream.error(' ===> File hazard "' + file_path_hazard +
                                                                 '" is not available. Check your folder.')
                                                raise FileNotFoundError('File not found.')

                                        elif self.scenario_tiling == 'weighted':

                                            file_path_hazard_right = self.define_file_hazard(
                                                self.folder_name_hazard, self.file_name_hazard,
                                                domain_name_step, section_scenario_tr_right_select)
                                            file_path_hazard_left = self.define_file_hazard(
                                                self.folder_name_hazard, self.file_name_hazard,
                                                domain_name_step, section_scenario_tr_left_select)

                                            file_data_hazard_right = read_file_hazard(
                                                file_path_hazard_right,
                                                file_vars=[self.domain_scenario_hazard_name],
                                                file_format=[self.domain_scenario_hazard_format],
                                                file_scale_factor=[self.domain_scenario_hazard_scale_factor])
                                            file_data_hazard_left = read_file_hazard(
                                                file_path_hazard_left,
                                                file_vars=[self.domain_scenario_hazard_name],
                                                file_format=[self.domain_scenario_hazard_format],
                                                file_scale_factor=[self.domain_scenario_hazard_scale_factor])

                                            if file_data_hazard_right is not None:
                                                file_data_h_right = file_data_hazard_right[self.domain_scenario_hazard_name]
                                            else:
                                                log_stream.error(' ===> File hazard "' + file_data_hazard_right +
                                                                 '" is not available. Check your folder.')
                                                raise FileNotFoundError('File not found.')

                                            if file_data_hazard_left is not None:
                                                file_data_h_left = file_data_hazard_left[self.domain_scenario_hazard_name]
                                            else:
                                                log_stream.error(' ===> File hazard "' + file_data_hazard_right +
                                                                 '" is not available. Check your folder.')
                                                raise FileNotFoundError('File not found.')

                                            # DEBUG
                                            # test_left = file_data_h_left[section_db_idx_hydraulic[0], section_db_idx_hydraulic[1]]
                                            # test_right = file_data_h_right[section_db_idx_hydraulic[0], section_db_idx_hydraulic[1]]

                                            file_data_h = (file_data_h_right * section_scenario_wg_right_check +
                                                           file_data_h_left * section_scenario_wg_left_check) / 2
                                            file_data_h[file_data_h < 0.0] = 0.0

                                            section_scenario_tr_select = (section_scenario_tr_right_select +
                                                                          section_scenario_tr_left_select) / 2

                                        else:
                                            log_stream.info(' ===> Scenario tiling method "' +
                                                            self.scenario_tiling + '" is not supported')
                                            raise NotImplementedError('Case not implemented yet')

                                        idx_x = section_area_idx[:, 0]
                                        idx_y = section_area_idx[:, 1]

                                        if section_db_idx_hydraulic is not None:
                                            section_scenario_h_select = file_data_h[section_db_idx_hydraulic[0], section_db_idx_hydraulic[1]]
                                        else:
                                            section_scenario_h_select = np.nan

                                        if not os.path.exists(file_path_scenario_anc_map):

                                            domain_scenario_merged_filled = deepcopy(domain_scenario_merged_default)

                                            file_data_h_scenario = file_data_h[idx_x, idx_y]

                                            domain_scenario_merged_filled[idx_x, idx_y] = file_data_h_scenario
                                            domain_scenario_merged_filled[domain_scenario_merged_filled <= 0] = np.nan

                                            folder_name, file_name = os.path.split(file_path_scenario_anc_map)
                                            make_folder(folder_name)

                                            if file_path_scenario_anc_map.endswith('tiff') or \
                                                    file_path_scenario_anc_map.endswith('tif'):

                                                save_file_tiff(file_path_scenario_anc_map,
                                                               domain_scenario_merged_filled,
                                                               domain_geo_x, domain_geo_y,
                                                               file_epsg_code='EPSG:32632')

                                            elif file_path_scenario_anc_map.endswith('workspace'):
                                                write_obj(file_path_scenario_anc_map, domain_scenario_merged_filled)
                                            else:
                                                log_stream.error(' ===> Save selected method is not supported.')
                                                raise NotImplementedError('Case not implemented yet')

                                        else:

                                            if file_path_scenario_anc_map.endswith('tiff') or \
                                                    file_path_scenario_anc_map.endswith('tif'):
                                                domain_scenario_merged_tmp = read_file_tiff(file_path_scenario_anc_map)
                                            elif file_path_scenario_anc_map.endswith('workspace'):
                                                domain_scenario_merged_tmp = read_obj(file_path_scenario_anc_map)
                                            else:
                                                log_stream.error(' ===> Read selected method is not supported.')
                                                raise NotImplementedError('Case not implemented yet')

                                            file_data_h_scenario = file_data_h[idx_x, idx_y]

                                            domain_scenario_merged_tmp[idx_x, idx_y] = file_data_h_scenario
                                            domain_scenario_merged_tmp[domain_scenario_merged_tmp <= 0] = np.nan

                                            if os.path.exists(file_path_scenario_anc_map):
                                                os.remove(file_path_scenario_anc_map)

                                            if file_path_scenario_anc_map.endswith('tiff') or \
                                                    file_path_scenario_anc_map.endswith('tif'):

                                                save_file_tiff(file_path_scenario_anc_map,
                                                               domain_scenario_merged_tmp,
                                                               domain_geo_x, domain_geo_y,
                                                               file_epsg_code='EPSG:32632')

                                            elif file_path_scenario_anc_map.endswith('workspace'):
                                                write_obj(file_path_scenario_anc_map, domain_scenario_merged_tmp)
                                            else:
                                                log_stream.error(' ===> Save selected method is not supported.')
                                                raise NotImplementedError('Case not implemented yet')

                                        if section_scenario_time not in list(file_path_scenarios_collections.keys()):
                                            file_path_scenarios_collections[section_scenario_time] = file_path_scenario_anc_map

                                        log_stream.info(' ------> Time step "' +
                                                        section_scenario_time.strftime(time_format_algorithm) +
                                                        '" ... DONE')

                                    else:
                                        section_scenario_tr_select = np.nan
                                        section_scenario_h_select = np.nan
                                        section_scenario_tr_right_select = np.nan
                                        section_scenario_tr_left_select = np.nan
                                        log_stream.info(
                                            ' ------> Time step "' +
                                            section_scenario_time.strftime(time_format_algorithm) +
                                            '" ... SKIPPED. Scenarios threshold is less then minimum threshold')

                                    section_scenario_time_list.append(section_scenario_time)
                                    section_scenario_discharge_list.append(section_scenario_discharge)
                                    section_scenario_tr_list.append(section_scenario_tr_select)
                                    section_scenario_h_list.append(section_scenario_h_select)
                                    section_scenario_tr_list_right.append(section_scenario_tr_right_select)
                                    section_scenario_tr_list_left.append(section_scenario_tr_left_select)

                                    section_scenario_tmp = list(zip(
                                        section_scenario_time_list, section_scenario_tr_list,
                                        section_scenario_discharge_list, section_scenario_h_list))

                                # Store time-series of tr, q and h (for a selected point)
                                dframe_data = {self.dframe_columns[0]: section_scenario_tr_list,
                                               self.dframe_columns[1]: section_scenario_discharge_list,
                                               self.dframe_columns[2]: section_scenario_h_list}
                                dframe_index = section_scenario_time_list

                                dframe_summary = pd.DataFrame(index=dframe_index, data=dframe_data)

                                section_dframe_collections[section_scenario_key] = dframe_summary
                                log_stream.info(' -----> Section "' + section_scenario_key + '" ... DONE')

                            else:
                                log_stream.info(' -----> Section "' + section_scenario_key +
                                                '" ... SKIPPED. Section datasets are empty')

                        else:
                            log_stream.info(' -----> Section "' + section_scenario_key +
                                            '" ... SKIPPED. Section info are empty')

                    # Save scenario maps file
                    folder_name_scenario_anc_domain_file, file_name_scenario_anc_domain_file = os.path.split(
                        file_path_scenario_anc_domain_file)
                    make_folder(folder_name_scenario_anc_domain_file)

                    file_data_scenario = {'file_path': file_path_scenarios_collections,
                                          'file_dframe': section_dframe_collections}
                    write_obj(file_path_scenario_anc_domain_file, file_data_scenario)

                    log_stream.info(' ----> Domain "' + domain_name_step + '" ... DONE')

                else:

                    # Load scenario maps file
                    file_data_scenario = read_obj(file_path_scenario_anc_domain_file)
                    file_path_scenarios_collections = file_data_scenario['file_path']
                    section_dframe_collections = file_data_scenario['file_dframe']

                    log_stream.info(' ----> Domain "' + domain_name_step + '" ... SKIPPED. File previously saved.')

                # Update collection workspace
                scenario_map_collection[domain_name_step] = file_path_scenarios_collections
                scenario_dframe_collection[domain_name_step] = section_dframe_collections

            else:
                log_stream.info(' ----> Domain "' + domain_name_step + '" ... SKIPPED. Domain datasets are empty')
                scenario_map_collection[domain_name_step] = None
                scenario_dframe_collection[domain_name_step] = None

        log_stream.info(' ---> Compute scenario maps [' + time.strftime(time_format_algorithm) + '] ... DONE')

        return scenario_map_collection, scenario_dframe_collection
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Method to organize scenario datasets
    def organize_scenario_datasets(self):

        time = self.time_run
        discharge_data_collection = self.discharge_data_collection
        geo_data_collection = self.geo_data_collection

        file_path_scenario_anc_collections_info = self.file_path_scenario_anc_info
        file_path_scenario_anc_collections_file = self.file_path_scenario_anc_file

        log_stream.info(' ---> Organize scenario datasets [' + time.strftime(time_format_algorithm) + '] ... ')

        scenario_info_collection = {}
        for domain_name_step in self.domain_name_list:

            log_stream.info(' ----> Domain "' + domain_name_step + '" ... ')

            domain_obj_data = discharge_data_collection[domain_name_step]
            domain_geo_data = geo_data_collection[domain_name_step]

            domain_discharge_index = geo_data_collection[domain_name_step][self.domain_discharge_index_tag]
            domain_grid_rows = geo_data_collection[domain_name_step][self.domain_grid_x_tag].shape[0]
            domain_grid_cols = geo_data_collection[domain_name_step][self.domain_grid_y_tag].shape[1]
            domain_section_db = geo_data_collection[domain_name_step][self.domain_sections_db_tag]

            file_path_scenario_anc_domain_info = file_path_scenario_anc_collections_info[domain_name_step]
            file_path_scenario_anc_domain_file = file_path_scenario_anc_collections_file[domain_name_step]

            if self.flag_cleaning_anc_scenario_info or self.flag_cleaning_anc_scenario_file:
                if os.path.exists(file_path_scenario_anc_domain_info):
                    os.remove(file_path_scenario_anc_domain_info)

            if not os.path.exists(file_path_scenario_anc_domain_info):
            
                domain_scenario_workspace = {}
                for section_obj_key, section_obj_dframe in domain_obj_data.items():

                    log_stream.info(' -----> Section "' + section_obj_key + '" ... ')

                    section_db_data = None
                    for domain_section_key, domain_section_fields in domain_section_db.items():
                        if domain_section_fields['description'] == section_obj_key:
                            section_db_data = domain_section_fields.copy()
                            break

                    if (section_db_data is not None) and (section_obj_dframe is not None):

                        section_discharge_data = section_obj_dframe[self.var_name_discharge]
                        section_type_data = section_obj_dframe[self.var_name_type]

                        section_db_n = section_db_data['n']
                        section_db_description = section_db_data['description']
                        section_db_name_outlet = section_db_data['name_point_outlet']
                        section_db_name_obs = section_db_data['name_point_obs']
                        section_db_idx = section_db_data['idx_data_terrain']

                        assert section_db_description == section_obj_key

                        # Compute scenario idx
                        section_discharge_idx = domain_discharge_index[section_db_idx[0] - 1, section_db_idx[1] - 1]

                        if self.scenario_analysis == 'max_period':

                            # Compute discharge for evaluating scenario
                            section_discharge_run, section_discharge_time, \
                                section_discharge_value, section_type_value, \
                                section_n_value = self.compute_scenario_discharge(
                                    section_discharge_data, section_type_data, analysis_freq=self.scenario_analysis)
                            # Get discharge attr(s)
                            section_discharge_attrs = section_discharge_data.attrs

                            # Compute tr for evaluating scenario
                            section_scenario_trs,\
                                section_scenario_trs_right, section_scenario_trs_left, \
                                section_scenario_weights_right, section_scenario_weights_left\
                                = self.compute_scenario_tr(
                                    section_discharge_idx, section_discharge_time, section_discharge_value,
                                    section_discharge_name=section_db_description,
                                    section_scenario_tr_min=self.tr_min, section_scenario_tr_max=self.tr_max)

                            domain_scenario_workspace[section_obj_key] = {}
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_index_tag] = section_scenario_trs
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_index_right_tag] = section_scenario_trs_right
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_index_left_tag] = section_scenario_trs_left
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_weight_right_tag] = section_scenario_weights_right
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_weight_left_tag] = section_scenario_weights_left
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_discharge_tag] = section_discharge_values
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_type_tag] = section_type_values
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_time_tag] = section_discharge_times
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_n_tag] = section_n_values
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_attrs_tag] = section_discharge_attrs
                            log_stream.info(' -----> Section "' + section_obj_key + '" ... DONE')

                        elif self.scenario_analysis == 'all_period':

                            # Compute discharge for evaluating scenario
                            section_discharge_runs, section_discharge_times, \
                                section_discharge_values, section_type_values, \
                                section_n_values = self.compute_scenario_discharge(
                                    section_discharge_data, section_type_data, analysis_freq=self.scenario_analysis)
                            # Get discharge attr(s)
                            section_discharge_attrs = section_discharge_data.attrs

                            # Compute tr for evaluating scenario
                            section_scenario_trs,\
                                section_scenario_trs_right, section_scenario_trs_left, \
                                section_scenario_weights_right, section_scenario_weights_left\
                                = self.compute_scenario_tr(
                                    section_discharge_idx, section_discharge_times, section_discharge_values,
                                    section_discharge_name=section_db_description,
                                    section_scenario_tr_min=self.tr_min, section_scenario_tr_max=self.tr_max)

                            section_tmp = list(zip(section_discharge_values, section_scenario_trs, section_scenario_trs_right, section_scenario_trs_left))

                            domain_scenario_workspace[section_obj_key] = {}
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_index_tag] = section_scenario_trs
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_index_right_tag] = section_scenario_trs_right
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_index_left_tag] = section_scenario_trs_left
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_weight_right_tag] = section_scenario_weights_right
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_weight_left_tag] = section_scenario_weights_left
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_discharge_tag] = section_discharge_values
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_type_tag] = section_type_values
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_time_tag] = section_discharge_times
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_n_tag] = section_n_values
                            domain_scenario_workspace[section_obj_key][self.domain_scenario_attrs_tag] = section_discharge_attrs

                            log_stream.info(' -----> Section "' + section_obj_key + '" ... DONE')

                        else:

                            log_stream.error(' ===> Scenario frequency value "' + str(self.scenario_analysis) +
                                             '" is not allowed')
                            log_stream.info(' -----> Section "' + section_obj_key + '" ... FAILED')
                            raise NotImplementedError('Case not implemented yet')

                    else:

                        log_stream.info(' -----> Section "' + section_obj_key + '" ... SKIPPED. Datasets are empty')
                        domain_scenario_workspace[section_obj_key] = None

                # Save scenario information file
                folder_name_scenario_anc_domain_info, file_name_scenario_anc_domain_info = os.path.split(
                    file_path_scenario_anc_domain_info)
                make_folder(folder_name_scenario_anc_domain_info)
                write_obj(file_path_scenario_anc_domain_info, domain_scenario_workspace)

                log_stream.info(' ----> Domain "' + domain_name_step + '" ... DONE')

            else:

                # Load scenario information file
                domain_scenario_workspace = read_obj(file_path_scenario_anc_domain_info)
                log_stream.info(' ----> Domain "' + domain_name_step + '" ... SKIPPED. File previously saved.')

            # Update collection workspace
            scenario_info_collection[domain_name_step] = domain_scenario_workspace

        log_stream.info(' ---> Organize scenario datasets [' + time.strftime(time_format_algorithm) + '] ... DONE')

        return scenario_info_collection

    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
