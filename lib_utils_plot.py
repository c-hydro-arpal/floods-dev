"""
Library Features:

Name:          lib_utils_plot
Author(s):     Fabio Delogu (fabio.delogu@cimafoundation.org)
Date:          '20220208'
Version:       '1.0.0'
"""

#######################################################################################
# Libraries
import logging
import json
import cartopy
import rasterio
import numpy as np
import pandas as pd

from lib_utils_io import write_file_tif, read_file_tif
from lib_utils_colormap import load

from copy import deepcopy

import matplotlib.pylab as plt
import matplotlib.ticker as mticker
import cartopy.io.img_tiles as cimgt

from pyproj import Proj

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

logging.getLogger('rasterio').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from lib_info_args import logger_name

# Logging
log_stream = logging.getLogger(logger_name)
#######################################################################################


# -------------------------------------------------------------------------------------
# Method to read data values from geotiff format
def read_file_tiff(file_name):
    file_data, file_proj, file_geotrans = read_file_tif(file_name)
    return file_data
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to save data values in geotiff format
def save_file_tiff(file_name, file_data, file_geo_x, file_geo_y, file_metadata=None, file_epsg_code='EPSG:32632'):

    if file_metadata is None:
        file_metadata = {'description': 'data'}

    file_data_height, file_data_width = file_data.shape

    file_geo_x_west = np.min(file_geo_x)
    file_geo_x_east = np.max(file_geo_x)
    file_geo_y_south = np.min(file_geo_y)
    file_geo_y_north = np.max(file_geo_y)

    file_data_transform = rasterio.transform.from_bounds(
        file_geo_x_west, file_geo_y_south, file_geo_x_east, file_geo_y_north,
        file_data_width, file_data_height)

    if not isinstance(file_data, list):
        file_data = [file_data]

    write_file_tif(file_name, file_data,
                   file_data_width, file_data_height, file_data_transform, file_epsg_code,
                   file_metadata=file_metadata)

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to save data values in png format
def save_file_png(file_name, file_data, file_geo_x, file_geo_y,
                  scenario_name='NA',
                  scenario_time_now_string='NA', scenario_time_step_string='NA',
                  fig_color_map_type=None, fig_dpi=150):

    if fig_color_map_type is None:
        fig_color_map_type = 'Blues'
    fig_color_map_obj = load(fig_color_map_type)

    p = Proj(proj='utm', zone=32, ellps='WGS84')
    file_lons, file_lats = p(file_geo_x, file_geo_y, inverse=True)

    file_lon_west = np.min(file_lons)
    file_lon_east = np.max(file_lons)
    file_lat_south = np.min(file_lats)
    file_lat_north = np.max(file_lats)

    plot_crs = cartopy.crs.Mercator()
    data_crs = cartopy.crs.PlateCarree()

    # Define graph title
    figure_title = ' == Floods Scenario - Catchment: ' + scenario_name + ' \n' \
                   ' TimeNow: ' + scenario_time_now_string + ' :: TimeStep: ' + scenario_time_step_string + ' == '

    # Create a Stamen Terrain instance.
    # map_background = cimgt.Stamen('terrain-background')
    # map_background = cimgt.OSM()
    # map_background = cimgt.GoogleTiles()
    map_background = cimgt.GoogleTiles(style='satellite')

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=plot_crs)
    ax.set_title(figure_title, size=12, color='black', weight='bold')
    # ax.coastlines(resolution='10m', color='black')
    ax.stock_img()
    ax.set_extent([file_lon_west, file_lon_east, file_lat_south, file_lat_north])

    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': 'gray', 'weight': 'bold'}
    gl.ylabel_style = {'size': 8, 'color': 'gray', 'weight': 'bold'}

    # Add the Stamen data at zoom level 8.
    ax.add_image(map_background, 14)

    sc = ax.pcolormesh(file_lons, file_lats, np.flipud(file_data), zorder=3,
                       cmap=fig_color_map_obj, transform=data_crs)

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cb1 = plt.colorbar(sc, cax=ax_cb, extend='both')
    cb1.set_label('water level [m]', size=12, color='gray', weight='normal')
    cb1.ax.tick_params(labelsize=10, labelcolor='gray')

    fig.savefig(file_name, dpi=fig_dpi)

    plt.show()
    plt.close()

# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Method to save data info in json format
def save_file_json(file_name, file_data_dict, file_indent=4, file_sep=','):

    file_data_json = {}
    for file_key, file_value in file_data_dict.items():
        if isinstance(file_value, list):
            file_value = [str(i) for i in file_value]
            file_value = file_sep.join(file_value)
        elif isinstance(file_value, (int, float)):
            file_value = str(file_value)
        elif isinstance(file_value, str):
            pass
        elif isinstance(file_value, dict):
            file_tmp = {}
            for value_key, value_data in file_value.items():
                if isinstance(value_data, np.datetime64):
                    time_stamp = pd.to_datetime(str(value_data))
                    time_str = time_stamp.strftime('%Y-%m-%d %H:%M')
                    file_tmp[value_key] = time_str
                else:
                    file_tmp[value_key] = value_data
            file_value = deepcopy(file_tmp)
        else:
            logging.error(' ===> Error in getting datasets')
            raise RuntimeError('Datasets case not implemented yet')

        file_data_json[file_key] = file_value

    file_data = json.dumps(file_data_json, indent=file_indent, ensure_ascii=False, sort_keys=True)
    with open(file_name, "w", encoding='utf-8') as file_handle:
        file_handle.write(file_data)

    pass
# -------------------------------------------------------------------------------------
