"""
Library Features:

Name:          lib_utils_geo
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220208'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import pyproj
import numpy as np

from copy import deepcopy
from pysheds.grid import Grid

import rasterio
from rasterio.crs import CRS

from lib_utils_io import read_mat
from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
# Debug
# import matplotlib.pylab as plt
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Geographical lookup table
geo_lookup_table = {'Latdem': 'latitude', 'Londem': 'longitude',
                    'a2dArea': 'cell_area', 'a2dCelle': 'cell_n',
                    'a2dDem': 'altitude', 'a2dQindice': 'discharge_idx',
                    'a2iChoice': 'channel_network', 'a2iPunt': 'flow_directions'}
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read geographical file in mat format
def read_file_geo(file_name):
    file_ws = {}
    file_data = read_mat(file_name)
    for file_key, file_values in file_data.items():
        if file_key in list(geo_lookup_table.keys()):
            var_idx = list(geo_lookup_table.keys()).index(file_key)
            var_name = list(geo_lookup_table.values())[var_idx]
            file_ws[var_name] = file_values

    return file_ws
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to read drainage area file in mat format
def read_file_drainage_area(file_name, file_excluded_keys=None):

    if file_excluded_keys is None:
        file_excluded_keys = ['__header__', '__version__', '__globals__', '__function_workspace__',
                              'Lat_dominio_UTM32', 'Lon_dominio_UTM32', 'None']

    file_ws = {}
    file_data = read_mat(file_name)
    for file_key, file_values in file_data.items():
        if file_key not in file_excluded_keys:
            file_ws[file_key] = file_values

    return file_ws
# -------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Method to define section mask
def compute_section_mask(fdir_values, fdir_map=None, fdir_nodata=0,
                         terrain_values=None, terrain_geo_x=None, terrain_geo_y=None,
                         terrain_epsg=4326, section_idx=None):

    if fdir_map is None:
        fdir_map = [8, 9, 6, 3, 2, 1, 4, 7]

    terrain_geo_x_min = np.min(terrain_geo_x)
    terrain_geo_x_max = np.max(terrain_geo_x)
    terrain_geo_y_min = np.min(terrain_geo_y)
    terrain_geo_y_max = np.max(terrain_geo_y)

    terrain_geo_width = terrain_geo_x.shape[1]
    terrain_geo_height = terrain_geo_y.shape[0]

    terrain_geo_x_res = (terrain_geo_x_max - terrain_geo_x_min) / (terrain_geo_width - 1)
    terrain_geo_y_res = (terrain_geo_y_max - terrain_geo_y_min) / (terrain_geo_height - 1)

    terrain_geo_affine = rasterio.transform.from_bounds(
        terrain_geo_x_min, terrain_geo_y_min, terrain_geo_x_max, terrain_geo_y_max,
        terrain_geo_width, terrain_geo_height)

    terrain_crs = CRS.from_epsg(terrain_epsg)

    mask_values = np.zeros([fdir_values.shape[0], fdir_values.shape[1]], dtype=bool)
    mask_values[:, :] = True

    grid = Grid()
    grid.add_gridded_data(data=fdir_values, data_name='fdir',
                          affine=terrain_geo_affine,
                          crs=pyproj.Proj(terrain_crs),
                          nodata=fdir_nodata)
    grid.add_gridded_data(data=mask_values, data_name='mask',
                          affine=terrain_geo_affine,
                          crs=pyproj.Proj(terrain_crs),
                          nodata=False)

    section_j = section_idx[0] - 1
    section_i = section_idx[1] - 1

    grid.catchment(data=grid.fdir, x=section_i, y=section_j,
                   dirmap=fdir_map, out_name='section_mask',
                   recursionlimit=15000, nodata_out=0, ytype='index')
    section_mask = np.array(grid.section_mask).astype(np.float32)
    section_transform = deepcopy(terrain_geo_affine)

    section_mask[section_mask == 0] = 0
    section_mask[terrain_values < 0] = 0
    section_mask[section_mask >= 1] = 1

    return section_mask, section_transform

# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Method to define section area
def compute_section_area(mask_map, mask_tranform,
                         terrain_values=None, terrain_geo_x=None, terrain_geo_y=None,
                         area_map_units='Km^2', area_map_rounding=1):

    area_cell_values = compute_cell_area(terrain_geo_x, terrain_geo_y,
                                         np.abs(mask_tranform[0]), np.abs(mask_tranform[4]))

    import math
    cell_size_x = Deg2Km_2(np.abs(mask_tranform[0]))
    cell_size_y = Deg2Km_2(np.abs(mask_tranform[4]))

    area_cell_values = cell_size_y * cell_size_x

    area_map = np.sum(area_cell_values * mask_map)
    area_pixels_count = np.count_nonzero(mask_map)

    '''
    if area_map_units == 'Km^2':
        area_tmp = np.float32(area_map / 1000000)
        area_map = float(str(round(area_tmp, area_map_rounding)))
    elif area_map_units == 'm^2':
        pass
    else:
        logging.error(' ===> Area mask units are not allowed')
        raise IOError('Area mask units are wrongly defined. Check your settings.')
    '''
    return area_map, area_pixels_count

# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Method to convert decimal degrees to km
def Deg2Km_2(deg, lat=None):
    if lat is None:
        km = deg * 110.54
    else:
        km = deg * 111.32 * np.cos(np.deg2rad(lat))
    return km
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# Method to convert decimal degrees to km (2)
def Deg2Km(deg):
    # Earth radius
    dRE = 6378.1370
    km = deg * (np.pi * dRE) / 180;
    return km
# --------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to compute cell area in m^2
def compute_cell_area(geo_x, geo_y, cell_size_x, cell_size_y):

    # Method constant(s)
    r = 6378388  # (Radius)
    e = 0.00672267  # (Ellipsoid)

    # dx = (R * cos(lat)) / (sqrt(1 - e2 * sqr(sin(lat)))) * PI / 180
    dx_2d = (r * np.cos(np.abs(geo_y) * np.pi / 180)) / \
            (np.sqrt(1 - e * np.sqrt(np.sin(np.abs(geo_y) * np.pi / 180)))) * np.pi / 180
    # dy = (R * (1 - e2)) / pow((1 - e2 * sqr(sin(lat))),1.5) * PI / 180
    dy_2d = (r * (1 - e)) / np.power((1 - e * np.sqrt(np.sin(np.abs(geo_y) / 180))), 1.5) * np.pi / 180

    # area cell in m^2
    area_cell = ((dx_2d / (1 / cell_size_x)) * (dy_2d / (1 / cell_size_y)))  # [m^2]

    return area_cell

# -------------------------------------------------------------------------------------
