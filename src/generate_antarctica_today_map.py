#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:18:05 2021

run "python generate_antarctica_today_map.py --help" to see command-line options.

@author: mmacferrin
"""
import cartopy
import geopandas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from osgeo import gdal
import argparse
import os
import numpy
import PIL
import datetime
import re
import tempfile
import pickle

import read_NSIDC_bin_file
import write_NSIDC_bin_to_gtif
# import svgclip
from map_filedata import boundary_shapefile_reader, \
                         mountains_shapefile_name,  \
                         map_picklefile_dictionary, \
                         annual_maps_directory,     \
                         anomaly_maps_directory,    \
                         region_outline_shapefiles_dict

from tb_file_data import model_results_picklefile, \
                         model_results_dir,        \
                         daily_melt_plots_dir

from melt_array_picklefile import read_model_array_picklefile, \
                                  get_ice_mask_array

from compute_mean_climatology import create_partial_year_melt_anomaly_tif, \
                                     read_annual_melt_anomaly_tif

def main():
    """Do stuff I want to do here."""

    m = AT_map_generator(fill_pole_hole=False, filter_out_error_swaths=True, verbose=True)

    for fmt in ("png", "pdf", "svg"):
        m.generate_annual_melt_map(outfile_template="/home/mmacferrin/Dropbox/Research/Antarctica_Today/text/BAMS SoC 2021/R0_2020-2021_sum.{0}".format(fmt),
                                   fmt=fmt,
                                   year=2020,reset_picklefile=True,dpi=600)

        m.generate_anomaly_melt_map(outfile_template="/home/mmacferrin/Dropbox/Research/Antarctica_Today/text/BAMS SoC 2021/R0_2020-2021_anomaly.{0}".format(fmt),
                                    fmt=fmt,
                                    year=2020,reset_picklefile=True,dpi=600)

    # fig, ax = m.generate_daily_melt_map("../data/v2.5/antarctica_melt_S3B_2010-2020_20200129/antarctica_melt_20100101_S3B_20210129.bin",
    #                                     "../plots/v2.5/daily_maps/20100101_daily.png",
    #                                     dpi=300,
    #                                     melt_code_threshold=6,
    #                                     include_scalebar=True,
    #                                     include_mountains=True,
    #                                     include_date=True,
    #                                     include_legend=True,
    #                                     reset_picklefile=False)



    # fig, ax = m.generate_daily_melt_map("../data/v2.5/antarctica_melt_S3B_2010-2020_20200129/antarctica_melt_20100101_S3B_20210129.bin",
    #                           outfile = "../plots/v2.5/daily_maps/20100101_daily.jpg", dpi=150)

    # print (m._get_current_axes_position(ax))

    # for fmt in ("png", "svg"):
    #     m.generate_annual_melt_map(outfile_template="../plots/v2.5/annual_maps/R{1}_{0}-{3}." + fmt,
    #                                 region_number=0,
    #                                 year=2020,
    #                                 dpi=600,
    #                                 message_below_year="through 16 February")
    #                                 # reset_picklefile=True)
    #                                 # melt_end_mmdd=(12,31),
    #                                 # include_current_date_label=True)

    #     # m.generate_anomaly_melt_map(year="all", reset_picklefile=True)
    #     m.generate_anomaly_melt_map(outfile_template="../plots/v2.5/anomaly_maps/R{1}_{0}-{3}." + fmt,
    #                                 year=2020,
    #                                 region_number=0,
    #                                 message_below_year="through 16 February,\n relative to 1990-2020",
    #                                 dpi=600)
    #                                 # mmdd_of_year=(2,10),
    #                                 # reset_picklefile=True)

    # for melt_code in range(2,8+1):
    #     m.generate_cumulative_melt_map(outfile_template = "../plots/v2.5/annual_maps/{0}_region{1}_level{2}.jpg",
    #                                    melt_code_threshold=melt_code,
    #                                    year="all")
    #                                    # year=2015)


def read_and_parse_args():
    """Read and parse the command-line arguments."""
    # Should have map generation for:
        # - daily melt plot or cumulative plot (or combined)
        # - all-contintent or regional (default region 0 -- all of Antartica)
        # - custom size (?)
    parser = argparse.ArgumentParser(description="Generate a daily and/or cumulative melt map.")
    parser.add_argument("output_file", type=str, help="Output image file. Format will be determined by the extension. Can include up to 3 format codes {0} {1} {2} to insert the year, region_number, and melt_level of the map, respectively.")
    parser.add_argument("-map_type", type=str, default="annual", help="Type of map to make. 'daily', 'annual' or 'annual_anomaly'. Default annual if not 'melt_file' if provided, daily if one is.")
    parser.add_argument("-melt_file", type=str, help="Path to a single melt file (.bin), for daily melt maps. Ignored if map_type is 'annual' or 'annual_anomaly'.")
    parser.add_argument("-year", type=str, default="all", help="Year to map, for annual and annual_anomaly plots only. Ignored for daily plots. Years refer to the beginning of the melt season: i.e. '2019' means the 2019-2020 melt season.")
    parser.add_argument("-region", type=int, default=0, help="The region number 0-7. Default 0 (all of Antarctica). 1 is the Antarctic Peninsula, going clockwise.")
    parser.add_argument("-melt_code_threshold", type=int, default=8, help="The melt threshold code 2-8, used in v2.5 of the data. 8 is all-inclusive, 2 is most-restrictive. Default 8. Use -1 if you want to plat them all.")
    parser.add_argument("-dpi", type=int, default=150, help="Resolution of output figure, in dots-per-inch (DPI). Ignored if 'output_map' is ununsed. Default 150. (NOTE: This DPI is off by about a factor of 50%. Cartopy determines image size by map extent (rather than imager aspect set by us), and thus often puts a large blank boundary around the image that we crop off. So the final image may be less DPI than specified here. The solution, if it's not adequate resolution, is to either save as a vector image (SVG), or increase the DPI to compensate.")
    parser.add_argument("--omit_scalebar", action="store_true", default=False, help="Omit the 1000 km scale bar. Default if not set: include it.")
    parser.add_argument("--omit_mountains", action="store_true", default=False, help="Omit the mountains from the figure. Default if not set: include them.")
    parser.add_argument("--omit_date", action="store_true", default=False, help="Keep the date off the figure (DD MMM YYYY for daily files, XXXX-YYYY melt year for annual files. Default if not set: Include the date.")
    parser.add_argument("--omit_legend", action="store_true", default=False, help="Omit the legend. Default if not set: include a legend.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Increase output verbosity.")

    return parser.parse_args()


class AT_map_generator:
    """A class for generating both daily, annual, and annual-anomaly melt maps.

    Stores internal information for creating the maps in order to help facilitate
    ease and re-use of matplotlib base figures, and speed up execution.
    """

    def __init__(self, melt_array_picklefile = model_results_picklefile,
                       fill_pole_hole = True,
                       filter_out_error_swaths = True,
                       verbose=True
                       ):
        """Initialize the class."""
        self.melt_array_picklefile = melt_array_picklefile
        self.melt_array = None
        self.datetimes_dict = None

        # Options for reading and/or gap-filling the data.
        self.OPT_fill_pole_hole = fill_pole_hole
        self.OPT_filter_out_error_swaths = filter_out_error_swaths
        self.OPT_verbose = verbose

        # Containers to store the pickled data for each basemap figure and axes object.
        # Can generate these once and save them to a picklefile, both on disk and
        # virtually in order to reuse again and again. (Pickling in the only way
        # to actually create full copies of matplotlib figures without them interacting
        # with previous versions generated.)
        #
        # Dictionary keys are the region number, "0" for all of Antarctica. The
        # other regions (1-7) are defined in tb_file_data.py
        self.figure_buffer_dict = {"daily":None,
                                   "annual":None,
                                   "anomaly":None}

        # The color maps and boundarynorm objects for each map type.
        self.melt_cmap_dict = {"daily":None,
                               "annual":None,
                               "anomaly":None}

        self.melt_norm_dict = {"daily":None,
                               "annual":None,
                               "anomaly":None}

        # The cartopy projection used. Create once and store it here.
        self.SPS_projection = self._generate_SPS_projection()

        # The (x,y) location of the meshgrid coordinates. Populated the first time
        # that self._get_meshgrid_coordinates() is called.
        self.meshgrid_coords = None

        # The mountains shapefile is huge. If we've already read it, save it here for reuse.
        self.mountains_df = None

        self.layers_order_and_alpha_dict = self._define_map_layers_zorder_and_alphas()

        # When reading in the melt array picklefile, it helps to cache it here and not have to keep re-reading it.
        self.cached_melt_array = None
        self.cached_datetime_dict = None

    def _generate_SPS_projection(self):
        """Generate the South Polar Stereo projection used by cartopy."""
        return cartopy.crs.SouthPolarStereo(central_longitude=0, true_scale_latitude=-70, globe=None)

    def get_melt_array_picklefile_and_datetimes(self):
        """Get the melt array picklefile and datetimes dictionary."""
        if self.cached_melt_array is None or self.cached_datetime_dict is None:
            self.cached_melt_array, self.cached_datetime_dict = \
                read_model_array_picklefile(fill_pole_hole          = self.OPT_fill_pole_hole,
                                            filter_out_error_swaths = self.OPT_filter_out_error_swaths,
                                            verbose                 = self.OPT_verbose)

        return self.cached_melt_array, self.cached_datetime_dict

    def _define_map_layers_zorder_and_alphas(self):
        """Define the map layering, what z-order and alphas each layer will have.

        The defined layers (dictionary keys) are "mountains", "boundaries",
                                        "data", "legend", "scalebar", and "labels".

        Values are a 2-tuple of (z-order relative to other layers, alpha transparency)
        """
        d = {}

        d["mountains"]  = (2, 0.42)
        d["boundaries"] = (3, 0.70) # boundaries are the continent & ice shelf boundaries
        d["data"]       = (1, 1.00)
        d["outline"]    = (4, 0.80) # outline is the outline of a given region.
        d["legend"]     = (5, 1.00)
        d["scalebar"]   = (5, 1.00)
        d["labels"]     = (5, 1.00)

        return d

    def _read_melt_array_and_datetimes(self):
        """Get the melt data. If we've already read it, just return it.

        Otherwise read it from the file.
        """
        # If we haven't stored either of the data sets, read the file and set them both.
        # NOTE: If we want to update the data file with new data, use the
        # "melt_array_picklefile.py" module to do that first before running this code.
        if self.melt_array is None or self.datetimes_dict is None:
            # Otherwise, read it from the file.
            self.melt_array, self.datetimes_dict = \
                read_model_array_picklefile(\
                        picklefile              = self.melt_array_picklefile,
                        fill_pole_hole          = self.OPT_fill_pole_hole,
                        filter_out_error_swaths = self.OPT_filter_out_error_swaths,
                        verbose                 = self.OPT_verbose
                        )


        return self.melt_array, self.datetimes_dict

    def _save_figure_to_buffer(self, fig,
                                     map_type="daily",
                                     overwrite=False):
        """Save a figure to a temporary memory buffer.

        If the buffer is already being used, simply return if "overwrite" is False.
        Otherwise, overwrite the buffer (or fill it if it's empty).)

        The memory buffer is useful for re-using a fresh copy of a basemap figure
        even though the last copy (pulled from a picklefile) was already modified
        and can't be reused. This is a way of repeatedly copying the figure object for reuse.
        """
        map_type_lower = map_type.lower().strip()
        existing_buffer = self.figure_buffer_dict[map_type_lower]

        if (existing_buffer is None) or overwrite:
            new_buffer = tempfile.SpooledTemporaryFile(mode="wb")
            pickle.dump(fig, new_buffer)
            new_buffer.seek(0) # Set the file pointer back to zero
            # Store in the dictionary.
            self.figure_buffer_dict[map_type_lower] = new_buffer

    def _get_fig_from_buffer(self, map_type="daily"):
        """If a buffer exists that is storing a fig object, read and return it.

        If no buffer has been written there, return None.
        """
        map_type_lower = map_type.lower().strip()
        buffer = self.figure_buffer_dict[map_type_lower]

        if buffer is None:
            return None

        else:
            buffer.seek(0)
            fig = pickle.load(buffer)
            buffer.seek(0)

            return fig

    def _get_meshgrid_data_coordinates(self):
        """Get the x,y map coordinates of each data pixel."""
        if self.meshgrid_coords is None:
            # Get the data x,y locations.
            x_vector, y_vector = write_NSIDC_bin_to_gtif.retrieve_ssmi_grid_coords(N_or_S="S", gridsize_km=25)
            # the grids are in km, put in m
            x_vector = x_vector * 1000
            y_vector = y_vector * 1000

            grid_x, grid_y = numpy.meshgrid(x_vector, y_vector)

            self.meshgrid_coords = grid_x, grid_y

        return self.meshgrid_coords


    def _get_map_extent(self, region_number):
        """For a given region number, provide the SouthPolarStereo map bounding box."""
        a = numpy.array # Just shorthand to make the code below cleaner

        # 50 km buffer. Can make bigger if we want.
        buffer = 50000
        buffer_array = a([-buffer,+buffer,-buffer,+buffer])
        # These extends are the boundary-envelope extents of each shapefile, grapped from QGIS.
        # I suppose I could open the shapefiles and extract the extents for each, but this works for now.
        if region_number in region_outline_shapefiles_dict:
            return {0:a([-2650000, 2725000,-2125000, 2225000]) + buffer_array,
                    1:a([-2650000,-1475000,  300000, 1675000]) + buffer_array,
                    2:a([-1675000,  900000, -200000, 1650000]) + buffer_array,
                    3:a([ -775000, 2275000,  825000, 2225000]) + buffer_array,
                    4:a([  800000, 2725000, -625000, 1200000]) + buffer_array,
                    5:a([  675000, 2575000,-2125000, -100000]) + buffer_array,
                    6:a([-1200000, 1375000,-2100000,  275000]) + buffer_array,
                    7:a([-1975000, -800000,-1325000,  325000]) + buffer_array,
                    }[region_number]

        else:
            raise NotImplementedError("Region {0} doesn't exist or is not implemented yet.".format(region_number))



    def _get_mountains_geodataframe(self):
        """Return the geodatadrame that holds the mountains shapefile.

        Use the cached version if already read.
        """
        if self.mountains_df is None:
            if self.OPT_verbose:
                print("Reading", mountains_shapefile_name)
            self.mountains_df = geopandas.read_file(mountains_shapefile_name, crs=self.SPS_projection.proj4_init)

        return self.mountains_df

    def _generate_new_baseline_map_figure(self, region_number=0,
                                                map_type = "daily",
                                                include_mountains = True,
                                                include_scalebar  = True,
                                                include_legend    = True,
                                                include_region_name_if_not_0    = True,
                                                include_region_outline_if_not_0 = True,
                                                region_to_outline = None,
                                                save_to_picklefile= True):
        """Read all the data, genereate a new baseline map figure.

        The map will contain
            (1) the continent outline,
            (2) the mountains (if "include_mountains"),
            (3) the color scale legend (if "include_legend")
            (5) the region name if it's a sub-region (if "include_region_name_if_not_0")

        If "save_to_picklefile", save the "figure" it to a pickle object for later use.

        Return matplotlib (figure, axes) objects.
        """
        map_type_lower = map_type.strip().lower()
        if map_type_lower not in ("daily", "annual", "anomaly"):
            raise ValueError("Uknown map type '{0}'.".format(map_type))

        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1,1,1, projection=self.SPS_projection)

        # Set the geographic extent.
        ax.set_extent(self._get_map_extent(region_number), self.SPS_projection )

        # Set the outline box to zero (no box on the map)
        ax.spines['geo'].set_linewidth(0)

        # Plot the basemap outline of the continent.
        b_z, b_alpha = self.layers_order_and_alpha_dict["boundaries"]
        ax.add_geometries(boundary_shapefile_reader.geometries(),
                          self.SPS_projection,
                          facecolor='none', linewidth=0.18, edgecolor='black',
                          zorder=b_z, alpha=b_alpha)

        if include_scalebar:
            self._add_scalebar(ax, region_number=region_number)
        # if False:
            # sb_z, sb_alpha = self.layers_order_and_alpha_dict["scalebar"]
            # scale_bar(ax, location=(0.30, 0.02), length=1000, linewidth=1,
            #           zorder=sb_z, alpha=sb_alpha,
            #           text_kwargs={"fontsize":"x-small"})

        if include_mountains:
            mz, m_alpha = self.layers_order_and_alpha_dict["mountains"]

            df = self._get_mountains_geodataframe()

            ax.add_geometries(df['geometry'], crs=self.SPS_projection,
                              facecolor=(118/255., 104/255., 64/255.),
                              edgecolor='black', linewidth=0.04,
                              zorder=mz, alpha=m_alpha)

        if include_legend:
            if map_type_lower == "daily":
                self._draw_legend_for_daily_melt(ax)

            elif map_type_lower == "annual":
                self._draw_legend_for_annual_melt(fig, ax)

            elif map_type_lower == "anomaly":
                self._draw_legend_for_melt_anomaly(fig, ax)

            else:
                raise ValueError("Uknown map type '{0}'.".format(map_type))

        if include_region_name_if_not_0 and region_number != 0:
            # TODO: implement this for different sub-regions
            pass

        if (include_region_outline_if_not_0 and region_number != 0) or (region_to_outline != None):
            if region_to_outline is None:
                region_to_outline = region_number
            self._add_region_outline(ax, region_number=region_to_outline)

        if save_to_picklefile:
            fname = map_picklefile_dictionary[(map_type_lower, region_number)]
            f = open(fname, 'wb')
            pickle.dump(fig, f)
            f.close()
            if self.OPT_verbose:
                print(fname, "written.")


        return fig, ax

    def _read_baseline_map_picklefile(self, map_type="daily",
                                            region_number=0):
        """Read the picklefile containing the basemap object, return the figure,axes instances.

        Return None if the picklefile doesn't exist.

        Sets the current matplotlib axes to the figures/axes read here.
        """
        map_type_lower = map_type.lower().strip()

        fname = map_picklefile_dictionary[(map_type_lower, region_number)]
        if not os.path.exists(fname):
            return None, None

        if self.OPT_verbose:
            print("Reading", fname)

        # Read the picklefile
        f = open(fname, 'rb')
        fig = pickle.load(f)
        f.close()
        # Get the axes, should just be one panel here.
        ax = fig.axes[0]
        # Set the current axes to ax
        plt.sca(ax)

        return fig, ax


    def _interpolate_color_levels(self, colors, levels, increment=1):
        """To create a smooth color map, set the levels where you want particular colors.

        This will create a set of boundaries and colors at "increment" resolution going
        between each defined RGB color tuple. Returns the boundaries, offset on either side by
        0.5 * increment.

        NOTE: Increment should be as small or smaller than the smallest gap between
        defined color levels. Behavoir is untested if it isn't.

        Return
        ------
        - New interpolated boundaries
        - New interpolated colors
        """
        color_arrays = numpy.array(colors)

        # levels = levels

        new_levels = numpy.arange(levels[0], levels[-1]+increment, increment)

        new_colors = numpy.empty((new_levels.shape[0], color_arrays.shape[1]), dtype=float)
        boundaries = numpy.empty((new_levels.shape[0]+1), dtype=float)

        boundaries[0] = new_levels[0] - 0.5*increment
        boundaries[1:] = new_levels + 0.5*increment

        assert len(new_levels) == len(new_colors)

        ref_color_level = levels[0]
        ref_color_level_next = levels[1]
        ref_color_idx = 0
        ref_color = color_arrays[0,:]
        ref_color_next = color_arrays[1,:]

        for i,new_level in enumerate(new_levels):
            if new_level in levels:
                ref_color_idx = levels.index(new_level)
                ref_color_level = new_level
                ref_color = color_arrays[ref_color_idx, :]
                new_colors[i,:] = ref_color

                if i<(len(new_levels)-1):
                    ref_color_level_next = levels[ref_color_idx + 1]
                    ref_color_next = color_arrays[ref_color_idx + 1, :]

            else:
                new_colors[i,:] = ref_color + (ref_color_next - ref_color) * (new_level - ref_color_level) / float(ref_color_level_next - ref_color_level)

        return boundaries, new_colors

    def _get_colormap_colors_levels_boundaries_and_labels(self, map_type = "daily",
                                                                interpolate = False,
                                                                increment = 1):
        """Return equal-length lists of colormap RGB-colors, levels, boundaries (between levels), and map labels.

        None for a label indicates no entry should be written for that label on
        the legend, nor on the scalebar.
        """
        # Use lower-case version, strip off any unnecessary whitespace
        map_type_lower = map_type.lower().strip()
        if map_type_lower not in ("daily", "annual", "anomaly"):
            raise ValueError("Map Type '{0}' not recognized. Use 'daily', 'annual', or 'anomaly'.".format(map_type))

        if map_type_lower == "daily":
            grey = (0.8,0.8,0.8)
            ocean = (0.90,0.95,1)
            white = (1,1,1)
            pink = (1,0.7,0.7)

            colors= [ocean, grey, white, pink]
            boundaries = [-1.5, -0.5, .5, 1.5, 2.5]
            levels = [-1, 0, 1, 2]
            labels = [None, "No Data", "No Melt", "Melt"]

        elif map_type_lower == "annual":
            # Define the colors for the colormap.
            # Finish filling these in. Use RGB values from eyedrop in a photo-editing program.
            # Can tweak these RBG values to adjust the color-scale on the map.
            grey_neg1        = [c/255. for c in (204, 204, 204)]
            white_0          = [c/255. for c in (255, 255, 255)]
            lightblue_1      = [c/255. for c in (209, 239, 255)]
            blue_10          = [c/255. for c in ( 43, 131, 186)]
            bluegreen_20     = [c/255. for c in (128, 191, 172)]
            green_30         = [c/255. for c in (199, 233, 173)]
            yellow_40        = [c/255. for c in (255, 255, 191)]
            orange_50        = [c/255. for c in (254, 201, 128)]
            darkorange_60    = [c/255. for c in (241, 124,  74)]
            red_70           = [c/255. for c in (215,  25,  28)]
            maroon_80        = [c/255. for c in (112,  12,  14)]

            colors = [grey_neg1,
                      white_0,
                      lightblue_1,
                      blue_10,
                      bluegreen_20,
                      green_30,
                      yellow_40,
                      orange_50,
                      darkorange_60,
                      red_70,
                      maroon_80]

            # Boundaries around the cutoffs.
            boundaries = [-1.5, -0.5, 0.5, 1.5, 10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5]
            levels = [-1, 0, 1, 10, 20, 30, 40, 50, 60, 70, 80]
            labels = [None, "0", "1", "10", "20", "30", "40", "50", "60", "70", "80+"]

            # Interpolate these to get a smooth curve of colors.
            if interpolate:
                boundaries, colors = self._interpolate_color_levels(colors, levels, increment=increment)

        elif map_type_lower == "anomaly":
            # Define the colors for the colormap.
            # Finish filling these in. Use RGB values from eyedrop in a photo-editing program.
            # Can tweak these RBG values to adjust the color-scale on the map.
            grey_neg999     = [c/255. for c in (204, 204, 204)]
            darkblue_neg200 = [c/255. for c in (  0,   0, 127)] # Create a far-out -200 value just to keep it from interpolating out ot -999.
            darkblue_neg45  = [c/255. for c in (  0,   0, 127)]
            blue_neg30      = [c/255. for c in (  0,   0, 255)]
            lightblue_neg15 = [c/255. for c in (127, 127, 255)]
            white_0         = [c/255. for c in (255, 255, 255)]
            pink_15         = [c/255. for c in (255, 127, 127)]
            red_30          = [c/255. for c in (255,   0,   0)]
            darkred_45      = [c/255. for c in (127,   0,   0)]

            colors = [grey_neg999,
                      darkblue_neg200,
                      darkblue_neg45,
                      blue_neg30,
                      lightblue_neg15,
                      white_0,
                      pink_15,
                      red_30,
                      darkred_45]

            # Boundaries around the cutoffs.
            boundaries = [-999.5, -200.5, -45.5, -30.5, -15.5, -0.5, 0.5, 15.5, 30.5, 45.5]
            levels = [-999, -200, -45, -30, -15, 0, 15, 30, 45]
            labels = [None, None, "-45", "-30", "-15", "0", "15", "30", "45+"]

            # Interpolate these to get a smooth curve of colors.
            if interpolate:
                boundaries, colors = self._interpolate_color_levels(colors, levels, increment=increment)

        return colors, levels, boundaries, labels


    def _create_colormap_and_norm(self, map_type = "daily", interpolate=False, increment=1):
        """Generate the color map and norm for a particular map type.

        map_type can be "daily", "annual", or "anomaly".
        (For now, the "annual" and "anomaly" maps use the same color scale.)

        Return:
        ------
        - matplotlib.colors.ListedColormap object
        - matplotlib.colors.BoundaryNorm   object

        Both return values are used in matplotlib.axes.pcolormesh() plotting method.
        """
        # Use lower-case version, strip off any unnecessary whitespace
        map_type_lower = map_type.lower().strip()
        if map_type_lower not in ("daily", "annual", "anomaly"):
            raise ValueError("Map Type '{0}' not recognized. Use 'daily', 'annual', or 'anomaly'.".format(map_type))

        colors,    \
        levels,    \
        boundaries,\
        labels = self._get_colormap_colors_levels_boundaries_and_labels(map_type=map_type_lower,
                                                                        interpolate=interpolate,
                                                                        increment=increment)

        melt_cmap = mcolors.ListedColormap(colors)
        melt_norm = mcolors.BoundaryNorm(boundaries, melt_cmap.N, clip=True)

        return melt_cmap, melt_norm

    def _get_colormap_and_norm(self, map_type = "daily",
                                     interpolate=False,
                                     increment=1):
        """Retreive the color map and norm for a particular map type.

        map_type can be "daily", "annual", or "anomaly".
        (For now, the "annual" and "anomaly" maps use the same color scale.)

        If these already exist in memory, return them.
        If not, generate them and save them to object data, then return.

        Return
        ------
        - matplotlib.colors.ListedColormap object
        - matplotlib.colors.BoundaryNorm   object
        """
        map_type_lower = map_type.lower().strip()
        if map_type_lower not in ("daily", "annual", "anomaly"):
            raise ValueError("Map Type '{0}' not recognized. Use 'daily', 'annual', or 'anomaly'.".format(map_type))

        if self.melt_cmap_dict[map_type_lower] is None or self.melt_norm_dict[map_type_lower] is None:
            self.melt_cmap_dict[map_type_lower], \
            self.melt_norm_dict[map_type_lower] = \
                self._create_colormap_and_norm(map_type = map_type_lower,
                                               interpolate=interpolate,
                                               increment=increment)

        return self.melt_cmap_dict[map_type_lower], self.melt_norm_dict[map_type_lower]



    def read_baseline_map_figure_and_axes(self, map_type="daily",
                                                region_number=0,
                                                skip_picklefile=False,
                                                read_from_buffer_if_available=True):
        """Read the baseline map figure, axes instances from the saved dictionary object and return them.

        If the pickle objects don't exist in memory, read them from the picklefiles.

        If the picklefiles don't exist on disk, create them from scratch and save them to disk.

        If "skip_picklefile" is True (for instance, to generate a new figure
        with different options than the baseline map), then generate it from scratch.
        """
        map_type = map_type.lower().strip()
        if read_from_buffer_if_available and self.figure_buffer_dict[map_type] != None:
            fig = self._get_fig_from_buffer(map_type=map_type)
            ax = fig.axes[0]

        if not skip_picklefile:
            fig, ax = self._read_baseline_map_picklefile(map_type, region_number)

        if skip_picklefile or (fig is None) or (ax is None):
            fig, ax = self._generate_new_baseline_map_figure(region_number=region_number,
                                                             map_type=map_type,
                                                             save_to_picklefile = False)

        return fig, ax

    def _strip_empty_image_border(self, filename):
        """Strip empty (alpha=0) image borders from the cartopy image written to disk.

        For some reason, cartopy likes to put a large empty border on the image.
        After writing the file, call this to get rid of it.

        WILL SKIP FOR VECTOR FILE FORMATS (like SVG or EPS)
        """
        extension = os.path.splitext(filename)[-1].strip().lower()
        if extension in (".eps", ".emf", ".pdf", ".svg"):
            return

        im = PIL.Image.open(filename)

        if extension == ".svg":
            return
        # svgclip.py isn't working... can't seem to resolve the Rsvg namespace.
            # svgclip.clip(filename, filename, margin=0)
            # if self.OPT_verbose:
            #     print(filename, "trimmed.")

        else:
            bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))
            diff = PIL.ImageChops.difference(im, bg)
            diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()
            if bbox:
                im2 = im.crop(bbox)
                im2.save(filename)
                if self.OPT_verbose:
                    print(filename, "trimmed.")

        return

    def _get_date_from_filename(self, filename):
        """Take a binary or geotiff filename with an 8-digit date in it, return a datetime.date object.

        Return None if no date found.
        """
        match = re.search("\d{8}", filename)
        if match is None:
            return None

        dts = match.group()
        return datetime.date(year=int(dts[0:4]), month=int(dts[4:6]), day=int(dts[6:8]))


    def _add_scalebar(self, ax, region_number=0):
        """Add a scalebar with label to the axes."""
        # First, get the total x-range of the figure, so that we can
        # appropriately size our 1000 km bar as a fraction of the axes width.
        xmin, xmax, ymin, ymax = self._get_map_extent(region_number=region_number)
        # Compute the fraction of the total width the scalebar will be.
        # map extent is in meters, scale by 1000 for km.

        # The length we use for each region map:
        length_km = {0:1000,
                     1:400,
                     2:400,
                     3:400,
                     4:400,
                     5:400,
                     6:400,
                     7:400,
                    }[region_number]

        location = {0:(0.02, 0.02),
                    1:(0.02, 0.02),
                    2:(0.02, 0.02),
                    3:(0.02, 0.02),
                    4:(0.02, 0.02),
                    5:(0.02, 0.02),
                    6:(0.02, 0.02),
                    7:(0.02, 0.02),
                    }[region_number]

        scalebar_fraction = float( length_km * 1000 ) / float( xmax - xmin )

        sb_z, sb_alpha = self.layers_order_and_alpha_dict["scalebar"]

        # Plot the line in axes coordinates.
        ax.plot((location[0], location[0]+scalebar_fraction), (location[1], location[1]),
                color="black",
                linewidth=1.5,
                transform=ax.transAxes,
                zorder=sb_z,
                alpha=sb_alpha)

        # Put the label above it, center-justified horizontally, bottom-justified vertically
        label_txt = "{0} km".format(length_km)
        label_location_x = location[0] + scalebar_fraction*0.5
        label_location_y = location[1]

        ax.text(label_location_x, label_location_y, label_txt,
                fontsize="5",
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                zorder=sb_z,
                alpha=sb_alpha)

        return

    def _add_date_to_axes(self, ax, filename, location=(0.06, 0.93)):
        """Add the date to an axes. Get the date from the file name.

        Location is in axes coordinates.
        """
        date = self._get_date_from_filename(filename)

        date_str = date.strftime("%-d %b %Y")

        z, alpha = self.layers_order_and_alpha_dict["labels"]

        ax.text(location[0], location[1], date_str, fontsize="large",
                transform=ax.transAxes, zorder=z, alpha=alpha)

        return

    def _add_year_to_axes(self, ax,
                                year_str,
                                region_number=0,
                                message_below_year = None,
                                message_below_year_fontsize=6):
        """Add the year to the axes. Provide the string rather than the integer.

        Location is in axes coordinates.
        """
        location = {0:(0.06, 0.93),
                    1:(0.94, 0.93),
                    2:(0.06, 0.93),
                    3:(0.06, 0.93),
                    4:(0.06, 0.93),
                    5:(0.06, 0.93),
                    6:(0.06, 0.93),
                    7:(0.06, 0.93),
                    }[region_number]

        horizontalalignment = {0:"left",
                               1:"right",
                               2:"left",
                               3:"left",
                               4:"left",
                               5:"left",
                               6:"left",
                               7:"left",
                    }[region_number]

        z, alpha = self.layers_order_and_alpha_dict["labels"]

        ax.text(location[0], location[1], year_str,
                fontsize="x-large",
                transform=ax.transAxes,
                zorder=z,
                alpha=alpha,
                ha=horizontalalignment)

        if message_below_year != None:
            ax.text(location[0], location[1]-0.021, message_below_year,
                    fontsize=message_below_year_fontsize,
                    fontstyle="italic",
                    transform=ax.transAxes,
                    zorder=z,
                    alpha=alpha,
                    va="top",
                    ha=horizontalalignment)


        return

    def _add_region_outline(self, ax, region_number=0):
        """Add a black line outlinining the region.

        If region == 0, this will outline all the regions on the map.
        """
        region_shapefile = region_outline_shapefiles_dict[region_number]

        shp_reader = cartopy.io.shapereader.Reader(region_shapefile)

        o_z, o_alpha = self._define_map_layers_zorder_and_alphas()["outline"]

        ax.add_geometries(shp_reader.geometries(),
                          self.SPS_projection,
                          facecolor='none', linewidth=0.5, edgecolor='black',
                          zorder=o_z, alpha=o_alpha)

    def _add_date_span_to_axes(self, ax, current_date, location=(0.06, 0.90)):
        """For annual maps of partial date totals so far, add the date range.

        This will be placed just under the date span (2020-2021, for instance) in smaller text.
        """
        z, alpha = self.layers_order_and_alpha_dict["labels"]

        date_str = current_date.strftime("(through %-d %b %Y)")

        ax.text(location[0], location[1], date_str, fontsize="small",
                transform=ax.transAxes, zorder=z, alpha=alpha)

        return

    def _add_level_to_axes(self, ax, level_number, location=(0.10, 0.87)):
        """Add the melt level to the axes. Only applicable to version 2.5 data."""
        z, alpha = self.layers_order_and_alpha_dict["labels"]

        ax.text(location[0], location[1], "Level {0}".format(level_number),
                fontsize="large", transform=ax.transAxes, zorder=z, alpha=alpha)

    def _draw_legend_for_daily_melt(self, ax, location=(0.04, 0.10)):
        """Add a color legend to the daily melt map.

        Location specifies the lower-left corner of the color legend, in axes coordinates.
        """
        colors, levels, boundaries, labels = self._get_colormap_colors_levels_boundaries_and_labels(map_type="daily")
        # Get rid of colors and labels that are None (leave off legend)
        colors = [c for i,c in enumerate(colors) if labels[i] != None]
        labels = [l for l in labels if l != None]

        # Let's put "No Data" at the bottom rather than at the top.
        colors = colors[1:] + [colors[0]]
        labels = labels[1:] + [labels[0]]

        box_locations = [(location[0], location[1]+(0.035*i)) for i in range(len(colors))][::-1]

        text_locations = [(x+0.030, y-0.002) for x,y in box_locations]

        z, alpha = self.layers_order_and_alpha_dict["legend"]

        for bl, tl, color, label in zip(box_locations, text_locations, colors, labels):

            ax.add_patch(matplotlib.patches.Rectangle(bl, width=0.025, height=0.025,
                                                      linewidth=0.5,
                                                      edgecolor='black',
                                                      facecolor=color,
                                                      transform=ax.transAxes,
                                                      zorder=z,
                                                      alpha=alpha))

            ax.text(tl[0], tl[1], label,
                    fontsize="xx-small",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    zorder=z,
                    alpha=alpha)

        return

    def _draw_legend_for_annual_melt(self, fig,
                                           ax,
                                           location=(0.03, 0.08)):
        """Add a color key legend to the annual cumulative melt maps."""
        colors, levels, boundaries, labels = self._get_colormap_colors_levels_boundaries_and_labels(map_type="annual",
                                                                                                    interpolate=False)
        # Get rid of colors and labels that are None (leave off legend)
        colors = [c for i,c in enumerate(colors) if labels[i] != None]
        labels = [l for l in labels if l != None]

        box_height = 0.025
        box_width  = 0.025

        # Put the box locations from top to bottom (in the reverse order they are listed.)
        box_locations = [(location[0], location[1]+(box_height*i)) for i in range(len(colors))][::-1]

        text_locations = [(x+box_width+0.005, y-0.0023) for x,y in box_locations]

        z, alpha = self.layers_order_and_alpha_dict["legend"]

        for bl, tl, color, label in zip(box_locations, text_locations, colors, labels):

            ax.add_patch(matplotlib.patches.Rectangle(bl, width=box_width, height=box_height,
                                                      linewidth=0.5,
                                                      edgecolor='black',
                                                      facecolor=color,
                                                      transform=ax.transAxes,
                                                      zorder=z,
                                                      alpha=alpha))

            ax.text(tl[0], tl[1], label,
                    fontsize=5,
                    transform=ax.transAxes,
                    va="bottom",
                    zorder=z,
                    alpha=alpha)

        legend_title_text = "Melt Days"
        # Put the "Melt Days" title above the top box.
        title_loc = (box_locations[0][0], box_locations[0][1] + 0.025)
        ax.text(title_loc[0], title_loc[1], legend_title_text,
                fontsize=6,
                transform=ax.transAxes,
                weight="bold",
                va="bottom",
                ha="left",
                zorder=z,
                alpha=alpha)


        return

    def _draw_legend_for_melt_anomaly(self, fig,
                                            ax,
                                            location=(0.03, 0.08)):
        """Add a color key legend to the cumulative melt anomaly maps."""
        colors, levels, boundaries, labels = self._get_colormap_colors_levels_boundaries_and_labels(map_type="anomaly",
                                                                                                    interpolate=False)
        # Get rid of colors and labels that are None (leave off legend)
        colors = [c for i,c in enumerate(colors) if labels[i] != None]
        labels = [l for l in labels if l != None]

        box_height = 0.030
        box_width  = 0.025

        # Put the box locations from top to bottom (in the reverse order they are listed.)
        box_locations = [(location[0], location[1]+(box_height*i)) for i in range(len(colors))][::-1]

        text_locations = [(x+box_width+0.005, y-0.0015) for x,y in box_locations]

        z, alpha = self.layers_order_and_alpha_dict["legend"]

        for bl, tl, color, label in zip(box_locations, text_locations, colors, labels):

            ax.add_patch(matplotlib.patches.Rectangle(bl, width=box_width, height=box_height,
                                                      linewidth=0.5,
                                                      edgecolor='black',
                                                      facecolor=color,
                                                      transform=ax.transAxes,
                                                      zorder=z,
                                                      alpha=alpha))

            ax.text(tl[0], tl[1], label,
                    fontsize=5,
                    transform=ax.transAxes,
                    va="bottom",
                    zorder=z,
                    alpha=alpha)

        legend_title_text = "Melt\nanomaly\n "
        # Put the "Melt Days" title above the top box.
        title_loc = (box_locations[0][0], box_locations[0][1] + box_height + 0.002)
        ax.text(title_loc[0], title_loc[1], legend_title_text,
                fontsize=6,
                weight="bold",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                zorder=z,
                alpha=alpha)

        ax.text(title_loc[0], title_loc[1], "(days)",
                fontsize=5,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                zorder=z,
                alpha=alpha)



        return

    def _get_current_axes_position(self, ax=None):
        """Get the figure position of the current axes.

        If "ax" is not provided,
        use matplotlib.pyplot.gca() to get current axes.
        """
        if ax is None:
            ax = plt.gca()

        return ax.get_position()

    def _scale_DPI_by_axes_size(self, fig, ax, DPI):
        """Rescale DPI to make it accurate.

        Right now the DPI fed to matplotlib ends up low-balled because of the
        funny way that cartopy draws an artificial border around the image, which
        we clip off after saving. Whatever DPI is fed to a plotting function,
        increase it by the proportion of the figure-width to the axes-width,
        in order to give the final, cropped figure the approximate DPI that was requested.
        """
        f_w, _ = fig.get_size_inches()

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_w, _ = bbox.width, bbox.height

        return int(DPI * f_w / ax_w)

    def plot_test_image(self, infile, outfile):
        """Just get this working."""
        infile_ext = os.path.splitext(infile)[-1].lower()
        # If it's a GeoTiff
        if infile_ext == ".tif":
            ds = gdal.Open(infile, gdal.GA_ReadOnly)
            if ds is None:
                raise Exception("{0} not read correctly by GDAL.".format(infile))

            data_array = ds.GetRasterBand(1).ReadAsArray()

        # If it's a flat binary
        elif infile_ext == ".bin":
            data_array = read_NSIDC_bin_file.read_NSIDC_bin_file(infile, return_type=int, signed=True)

        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(1,1,1, projection=self.SPS_projection)

        # Set the geographic extent.
        ax.set_extent(self._get_map_extent(0), self.SPS_projection )

        # Set the outline box to zero (no box on the map)
        ax.spines['geo'].set_linewidth(0)

        # Plot the basemap outline of the continent.
        b_z, b_alpha = self.layers_order_and_alpha_dict["boundaries"]
        ax.add_geometries(boundary_shapefile_reader.geometries(),
                          self.SPS_projection,
                          facecolor='none', linewidth=0.18, edgecolor='black',
                          zorder=b_z, alpha=b_alpha)

        grid_x, grid_y = self._get_meshgrid_data_coordinates()

        melt_cmap, melt_norm = self._get_colormap_and_norm(map_type="daily")
        data_z, data_alpha = self.layers_order_and_alpha_dict["data"]

        import numpy
        print(numpy.unique(data_array))
        for i in range(-1,8+1):
            print(i, numpy.count_nonzero(data_array==i))

        # Plot the data
        ax.pcolormesh(grid_x, grid_y, data_array, transform=self.SPS_projection,
                      cmap=melt_cmap, norm=melt_norm,
                      zorder=data_z, alpha=data_alpha)

        self._draw_legend_for_daily_melt(ax)

        fig.savefig(outfile, dpi=150)
        print(outfile, "written.")

        self._strip_empty_image_border(outfile)

        return

    def _produce_melt_year_date_masks(self, datetimes_dict,
                                  year="all",
                                  melt_start_mmdd = (10,1),
                                  melt_end_mmdd = (4,30)):
        """Given a dictionary of (datetime:index) key:value pairs, return masks for all the years requested.

        year: "all", or an integer year (referring to the start year of the melt season)
        melt_start_mmdd: (mm,dd) tuple indicating the start of the melt season
        melt_end_mmdd: (mm,dd) tuple indicating the end of the melt season.

        Returns
        -------
            - List of years
            - List of 1-D boolean masks into the array.

        If a single year is input into the "year" parameter, each of the lists
        will be length 1.
        """
        datetimes_list = list(datetimes_dict.keys())
        datetimes_list.sort()

        if year == "all":
            unique_years = numpy.unique([dt.year for dt in datetimes_list])
            unique_years.sort()
        elif type(year) in (int, float, str):
            unique_years = [int(year)]
        elif type(year) in (list, tuple):
            unique_years = year

        year_list = []
        mask_list = []
        for iter_year in unique_years:
            start_date = datetime.datetime(year=iter_year, month=melt_start_mmdd[0], day=melt_start_mmdd[1])
            end_date = datetime.datetime(year=((iter_year+1) if melt_end_mmdd < melt_start_mmdd else iter_year),
                                         month = melt_end_mmdd[0], day = melt_end_mmdd[1])

            # Only append if there are any days included in that particular year.
            # This avoids "extra" years being added at the tail-end of the dataset that contain
            # no actual data.
            mask = numpy.array([(start_date <= dt <= end_date) for dt in datetimes_list], dtype=bool)
            if numpy.count_nonzero(mask) > 0:
                year_list.append(iter_year)
                mask_list.append(mask)

        return year_list, mask_list


    def generate_daily_melt_map(self, infile="latest",
                                      outfile=None,
                                      dpi=150,
                                      region_number=0,
                                      include_region_name_if_not_0=True,
                                      include_region_outline_if_not_0=True,
                                      region_to_outline=None,
                                      include_scalebar=True,
                                      include_legend=True,
                                      include_mountains=True,
                                      include_date=True,
                                      reset_picklefile=False):
        """Generate a daily melt map. Output to "outfile"."""
        # If infile == "latest", get the latest .bin file.
        if infile.strip().lower() == "latest":
            infile = os.path.join(model_results_dir, max(os.listdir(model_results_dir)))

        # 1: Read the data file into an array.
        if self.OPT_verbose:
            print ("Reading", infile)


        infile_ext = os.path.splitext(infile)[-1].lower()
        # If it's a GeoTiff
        if infile_ext == ".tif":
            ds = gdal.Open(infile, gdal.GA_ReadOnly)
            if ds is None:
                raise Exception("{0} not read correctly by GDAL.".format(infile))

            data_array = ds.GetRasterBand(1).ReadAsArray()

        # If it's a flat binary
        elif infile_ext == ".bin":
            data_array = read_NSIDC_bin_file.read_NSIDC_bin_file(infile, return_type=int, signed=True)

        # If we're using all the defaults, just read the baseline image from the picklefile.
        if include_region_name_if_not_0 == True and \
           include_scalebar == True and \
           include_legend == True and \
           include_mountains == True:
            fig, ax = self.read_baseline_map_figure_and_axes(map_type="daily",
                                                             region_number=region_number,
                                                             skip_picklefile=reset_picklefile)

        else:
            fig, ax = self._generate_new_baseline_map_figure(region_number=region_number,
                                                             map_type="daily",
                                                             include_mountains=include_mountains,
                                                             include_scalebar=include_scalebar,
                                                             include_legend=include_legend,
                                                             include_region_name_if_not_0=include_region_name_if_not_0,
                                                             include_region_outline_if_not_0=include_region_outline_if_not_0,
                                                             region_to_outline=region_to_outline,
                                                             save_to_picklefile=reset_picklefile)


        grid_x, grid_y = self._get_meshgrid_data_coordinates()

        melt_cmap, melt_norm = self._get_colormap_and_norm(map_type="daily")
        data_z, data_alpha = self.layers_order_and_alpha_dict["data"]

        # Plot the data
        ax.pcolormesh(grid_x, grid_y, data_array, transform=self.SPS_projection,
                      cmap=melt_cmap, norm=melt_norm,
                      zorder=data_z, alpha=data_alpha)

        if include_date:
            # TODO: Adjust to give region_number, not a specific location.
            self._add_date_to_axes(ax, infile, (0.06, 0.93))

        if outfile != None:
            if outfile.strip().lower() == "auto":
                dt = self._get_date_from_filename(os.path.split(infile)[1])
                outfile = os.path.join(daily_melt_plots_dir, dt.strftime("%Y.%m.%d.png"))

            new_dpi = self._scale_DPI_by_axes_size(fig, ax, dpi)
            fig.savefig(outfile, dpi=new_dpi)

            if self.OPT_verbose:
                print(outfile, "written.")

            self._strip_empty_image_border(outfile)

        return fig, ax

    def generate_annual_melt_map(self, outfile_template=None,
                                       year="all",
                                       fmt="png",
                                       melt_start_mmdd = (10,1),
                                       melt_end_mmdd = (4,30),
                                       dpi=150,
                                       region_number=0,
                                       include_region_name_if_not_0=True,
                                       include_region_outline_if_not_0=True,
                                       region_to_outline=None,
                                       include_scalebar=True,
                                       include_legend=True,
                                       include_mountains=True,
                                       include_year_label=True,
                                       keep_year_label_wrapped=True,
                                       gap_filled=False,
                                       message_below_year=None,
                                       reset_picklefile=False):
        """Generate a cumulative annual melt map. Write out "outfile".

        outfile_template can be a name of an output image file, but can also have
        up to three format codes "{0} {1} {2}" anywhere in the file name. Those
        codes will be filled with:
        - {0}: the melt year being used (2019, e.g.)
        - {1}: the region_number,
        - {2}: the *next* year (useful if you want to name it 2019-2020.tif, for instance.)

        These fields are only included in the output filename where the above codes
        are included in the strong "outfile_template". Any codes that are omitted
        will not include the respective fields in the output filename.
        This is useful for creating multiple files in different years, for instance.
        """
        # Read the melt data file into an array and datetime dictionary
        melt_array, datetime_dict = self.get_melt_array_picklefile_and_datetimes()

        years, masks = self._produce_melt_year_date_masks(datetime_dict,
                                                          year=year,
                                                          melt_start_mmdd=melt_start_mmdd,
                                                          melt_end_mmdd=melt_end_mmdd)

        # If we're using all the defaults, just read the baseline image from the picklefile.
        if not reset_picklefile and \
           include_region_name_if_not_0 == True and \
           include_scalebar == True and \
           include_legend == True and \
           include_mountains == True and \
           include_region_outline_if_not_0 == True and \
           ((region_to_outline is None) or (region_to_outline == region_number)):
            fig, ax = self.read_baseline_map_figure_and_axes(map_type="annual",
                                                             region_number=region_number,
                                                             skip_picklefile=reset_picklefile)

        else:
            fig, ax = self._generate_new_baseline_map_figure(region_number=region_number,
                                                             map_type="annual",
                                                             include_mountains=include_mountains,
                                                             include_scalebar=include_scalebar,
                                                             include_legend=include_legend,
                                                             include_region_name_if_not_0=include_region_name_if_not_0,
                                                             include_region_outline_if_not_0=include_region_outline_if_not_0,
                                                             region_to_outline=region_to_outline,
                                                             save_to_picklefile=reset_picklefile)

        # If we're generating more than one figure, save the figure to a memory
        # buffer for reuse.
        if len(years) > 1:
            self._save_figure_to_buffer(fig, map_type="annual", overwrite=False)

        grid_x, grid_y = self._get_meshgrid_data_coordinates()

        melt_cmap, melt_norm = self._get_colormap_and_norm(map_type="annual",
                                                           interpolate=True,
                                                           increment=1)
        data_z, data_alpha = self.layers_order_and_alpha_dict["data"]

        ice_mask = numpy.array(get_ice_mask_array(), dtype=bool)

        # Loop over each year/mask, pull out the data, sum up the days (at/under each threshold)
        for y,mask in zip(years, masks):

            if len(years) > 1:
                fig = self._get_fig_from_buffer(map_type="annual")
                ax = fig.axes[0]

            # Pull the data for just that year.
            # print(melt_array.shape, mask.shape, len(datetime_dict))
            year_slice = melt_array[:,:,mask]

            # Set the melt values above melt (2) but below the cutoff thresholds we've set in data version 2.5
            # This will still work in a v3 that doesn't have those codes, if all melt values == 2.
            boolean_melt_slice = (year_slice >= 2)

            # Sum up the number of melt days along the time axis
            cumulative_melt_slice = numpy.sum(boolean_melt_slice, axis=2)

            # Set regions outside the ice mask to NODATA (-1)
            cumulative_melt_slice[~ice_mask] = -1

            ax.pcolormesh(grid_x,
                          grid_y,
                          cumulative_melt_slice,
                          transform=self.SPS_projection,
                          cmap=melt_cmap,
                          norm=melt_norm,
                          zorder=data_z,
                          alpha=data_alpha)

            if include_year_label:
                if keep_year_label_wrapped or (melt_end_mmdd < melt_start_mmdd):
                    year_str = "{0:d}-{1:d}".format(y, y+1)
                else:
                    year_str = str(y)

                self._add_year_to_axes(ax, year_str, region_number=region_number, message_below_year=message_below_year)

            if outfile_template == None:
                outfile_template = os.path.join(annual_maps_directory, "R{1}_{0}-{2}." + ("png" if fmt is None else fmt))

            outfile_fname = outfile_template.format(y, region_number, y+1)

            new_dpi = self._scale_DPI_by_axes_size(fig, ax, dpi)
            fig.savefig(outfile_fname, dpi=new_dpi)

            if self.OPT_verbose:
                print(outfile_fname, "written.")

            self._strip_empty_image_border(outfile_fname)

        return fig, ax


    def generate_anomaly_melt_map(self, outfile_template=None,
                                        year="all",
                                        fmt="png",
                                        mmdd_of_year=None,
                                        melt_start_mmdd = (10,1),
                                        melt_end_mmdd = (4,30),
                                        dpi=150,
                                        region_number=0,
                                        include_region_name_if_not_0=True,
                                        include_region_outline_if_not_0=True,
                                        region_to_outline=None,
                                        include_scalebar=True,
                                        include_legend=True,
                                        include_mountains=True,
                                        include_year_label=True,
                                        keep_year_label_wrapped=True,
                                        reset_picklefile=False,
                                        message_below_year="relative to 1990-2020",
                                        verbose=True):
        """Generate a cumulative annual anomaly melt map compared to the baseline climatology period.

        Write out "outfile".

        Year can be a specific melt year (2019 for 2019-20, for instance), or "all" if you want to output them all.

        mmdd_of_year is a specific (mm,dd) tuple if you just want a partial-anomaly map
        for part of a year. Use "None" if you want the end-of-melt-season comparison.

        outfile_template can be a name of an output image file, but can also have
        up to three format codes "{0} {1} {2}" anywhere in the file name. Those
        codes will be filled with the melt year being used, the region_number,
        and the mmdd_of_year, respectively, if they are included in the
        output_template string. This is useful for creating multiple files in
        different years, for instance.
        """
        if year == "all":
            # Just get all the year up through present. It's okay if some are blank.
            years = range(1979, datetime.date.today().year + 1)
        else:
            assert year == int(year)
            years = [year]

        # If we're using all the defaults, just read the baseline image from the picklefile.
        if not reset_picklefile and \
           include_region_name_if_not_0 == True and \
           include_scalebar == True and \
           include_legend == True and \
           include_mountains == True and \
           include_region_outline_if_not_0 == True and \
           ((region_to_outline is None) or (region_to_outline == region_number)):

            fig, ax = self.read_baseline_map_figure_and_axes(map_type="anomaly",
                                                             region_number=region_number,
                                                             skip_picklefile=reset_picklefile)

        else:
            fig, ax = self._generate_new_baseline_map_figure(region_number=region_number,
                                                             map_type="anomaly",
                                                             include_mountains=include_mountains,
                                                             include_scalebar=include_scalebar,
                                                             include_legend=include_legend,
                                                             include_region_name_if_not_0=include_region_name_if_not_0,
                                                             include_region_outline_if_not_0=include_region_outline_if_not_0,
                                                             region_to_outline=region_to_outline,
                                                             save_to_picklefile=reset_picklefile)

        # If we're generating more than one figure, save the figure to a memory
        # buffer for reuse.
        if len(years) > 1:
            self._save_figure_to_buffer(fig, map_type="anomaly", overwrite=False)

        grid_x, grid_y = self._get_meshgrid_data_coordinates()

        melt_cmap, melt_norm = self._get_colormap_and_norm(map_type="anomaly",
                                                           interpolate=True,
                                                           increment=1)
        data_z, data_alpha = self.layers_order_and_alpha_dict["data"]

        for year in years:

            if len(years) > 1:
                fig = self._get_fig_from_buffer(map_type="anomaly")
                ax = fig.axes[0]

            if mmdd_of_year is None:
                # Just get the annual anomlay map for that year.
                anomaly_data = read_annual_melt_anomaly_tif(year=year,
                                                            verbose=verbose)
            else:
                datetime_this_year = datetime.datetime(year=year + (0 if mmdd_of_year >= melt_start_mmdd else 1),
                                                       month=mmdd_of_year[0],
                                                       day=mmdd_of_year[1])
                anomaly_data = create_partial_year_melt_anomaly_tif(current_datetime=datetime_this_year, gap_filled=False, verbose=verbose)

            if anomaly_data is None:
                continue

            ax.pcolormesh(grid_x,
                          grid_y,
                          anomaly_data,
                          transform=self.SPS_projection,
                          cmap=melt_cmap,
                          norm=melt_norm,
                          zorder=data_z,
                          alpha=data_alpha)

            if include_year_label:
                if keep_year_label_wrapped or (melt_end_mmdd < melt_start_mmdd):
                    year_str = "{0:d}-{1:d}".format(year, year+1)
                else:
                    year_str = str(year)

                self._add_year_to_axes(ax, year_str, region_number=region_number, message_below_year=message_below_year)


            if outfile_template == None:
                outfile_template = os.path.join(anomaly_maps_directory, "R{1}_{0}-{2}.png")

            if mmdd_of_year is None:
                outfile_fname = outfile_template.format(year, region_number, year+1)
            else:
                outfile_fname = outfile_template.format(year, region_number, datetime_this_year.strftime("%Y.%m.%d"), year+1)

            new_dpi = self._scale_DPI_by_axes_size(fig, ax, dpi)
            fig.savefig(outfile_fname, dpi=new_dpi)

            if self.OPT_verbose:
                print(outfile_fname, "written.")

            self._strip_empty_image_border(outfile_fname)

        return fig, ax

    def generate_latest_partial_anomaly_melt_map(self, outfile_template=None,
                                                       fmt="png",
                                                       dpi=150,
                                                       melt_start_mmdd = (10,1),
                                                       melt_end_mmdd = (4,30),
                                                       region_number=0,
                                                       include_region_name_if_not_0=True,
                                                       include_region_outline_if_not_0=True,
                                                       region_to_outline=None,
                                                       include_scalebar=True,
                                                       include_legend=True,
                                                       include_mountains=True,
                                                       include_year_label=True,
                                                       keep_year_label_wrapped=True,
                                                       reset_picklefile=False,
                                                       message_below_year=None,
                                                       verbose=True):
        """Same as generate_anomaly_melt_map, but do it for only a partial year,
        up until the last day of data that we have in the melt array.

        Uses the "mmdd_of_year" parameter in the .generate_anomaly_melt_map() method to do this.
        Just grabs the latest date in the array first."""
        melt_array, dates = self._read_melt_array_and_datetimes()
        # Get whatever the last date is
        date = sorted(dates.keys())[-1]
        year = date.year + (0 if ((date.month, date.day) >= melt_start_mmdd) else -1)
        mmdd_today = (date.month, date.day)

        self.generate_anomaly_melt_map(outfile_template=outfile_template,
                                       year=year,
                                       fmt=fmt,
                                       dpi=dpi,
                                       mmdd_of_year = mmdd_today,
                                       melt_start_mmdd = melt_start_mmdd,
                                       melt_end_mmdd = melt_end_mmdd,
                                       region_number = region_number,
                                       include_region_name_if_not_0 = include_region_name_if_not_0,
                                       include_region_outline_if_not_0= include_region_outline_if_not_0,
                                       region_to_outline=region_to_outline,
                                       include_scalebar = include_scalebar,
                                       include_legend = include_legend,
                                       include_mountains = include_mountains,
                                       include_year_label = include_year_label,
                                       keep_year_label_wrapped = keep_year_label_wrapped,
                                       reset_picklefile = reset_picklefile,
                                       message_below_year = message_below_year,
                                       verbose=verbose)


def SPECIAL_make_map_with_borders(year=2020):
    """Make a map specifically for the BAMS 2020-21 report.

    Includes a full melt map for 2019-20 (region 0, all continent), but with an
    outline overlaid of the "Antarctic Peninsula" region on the map.

    This takes a bit of tweaked-coding in the function.
    """
    region_shapefile = "../qgis/basins/Antarctic_Regions_v2_interior_borders.shp"

    at = AT_map_generator(fill_pole_hole=False, verbose=True)
    for fmt in ("png", "svg"):
    # for fmt in ("png",):
        fname = os.path.join(annual_maps_directory, "R0_{0}-{1}_region_borders_2021.02.16.".format(year,year+1) + fmt)
        fig, ax = at.generate_annual_melt_map(outfile_template = fname,
                                              region_number=0,
                                              year=year,
                                              message_below_year="through 16 February",
                                              dpi=600,
                                              region_to_outline=None,
                                              include_region_outline_if_not_0=False,
                                              reset_picklefile=False)

        shp_reader = cartopy.io.shapereader.Reader(region_shapefile)

        o_z, o_alpha = at._define_map_layers_zorder_and_alphas()["outline"]

        ax.add_geometries(shp_reader.geometries(),
                          at.SPS_projection,
                          facecolor='none', linewidth=0.45, edgecolor='black',
                          zorder=o_z, alpha=o_alpha)

        # TODO: Add text labels of region names.
        fs = 5.5 # Font size
        ax.text(0.1, 0.82, "Antarctic\nPeninsula", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)
        ax.text(0.45, 0.70, "Ronne\nEmbayment", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)
        ax.text(0.61, 0.84, "Maud and Enderby", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)
        ax.text(0.7, 0.6, "Amery and\nShackleton", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)
        ax.text(0.76, 0.28, "Wilkes and\nAdelie", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)
        ax.text(0.56, 0.4, "Ross\nEmbayment", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)
        ax.text(0.12, 0.23, "Amundsen\nBellinghausen", va="top", ha="left", fontsize=fs, fontweight="semibold",transform=ax.transAxes)

        dpi = at._scale_DPI_by_axes_size(fig, ax, 600)
        fig.savefig(fname, dpi=dpi)
        at._strip_empty_image_border(fname)

        print(fname, "overwritten.")

if __name__ == "__main__":
    main()
    # SPECIAL_make_map_with_borders()
    # generate_annual_melt_map()
