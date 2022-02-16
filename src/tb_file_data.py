"""Created on Wed Oct 14 15:19:20 2020.

Code for keeping track of Tb-based model result data, and other products

@author: mmacferrin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Get the list of .bin files in the v1 data
def recurse_directory(directory, ignore="thresholds", target=".bin", sorted=True):
    """Recurse into the directory, and find all .bin files. Return a list of them."""
    dir_list = os.listdir(directory)

    file_list = []
    for dname in dir_list:
        dpath = os.path.join(directory, dname)
        if os.path.isdir(dpath):
            if dname.find(ignore) == -1:
                file_list.extend(recurse_directory(dpath,
                                                   ignore=ignore,
                                                   target=target,
                                                   sorted=False)
                                 )

        else:
            if dname.find(target) >= 0:
                file_list.append(dpath)

    if sorted:
        file_list.sort()
    return file_list


gridded_elevation_bin_file = "../baseline_datasets/REMA_25km_resampled_TbGrid.bin"

# Tif file containing a gridded version assigning each pixel in the ice mask to a specific region.
# The dictionary gives the names of each region based on the value in the tif file.
antarctic_regions_tif = "../baseline_datasets/Antarctica_Regions_Combined_v2.tif"
antarctic_regions_dict = {0: "Antarctica",
                          1: "Antarctic Peninsula",
                          2: "Ronne Embayment",
                          3: "Maud and Enderby",
                          4: "Amery and Shackleton",
                          5: "Wilkes and Adelie",
                          6: "Ross Embayment",
                          7: "Amundsen Bellinghausen"}

# Save to a more-generic version of the array and picklefile lists, to use below

NSIDC_0080_file_dir = "../Tb/nsidc-0080"

DATA_VERSION = 3

if DATA_VERSION == 1:
    # Temporarily cut these out to reduce overhead when importing.
    # model_results_v1_file_list = recurse_files(model_results_v1_dir)
    # model_results_v1_file_list = None
    model_results_v1_dir = "../data/v1"

    model_results_v1_picklefile = os.path.join(model_results_v1_dir, "v1_2000-2020.pickle")
    model_results_v1_plot_directory = "../plots/v1"
    outputs_v1_tifs_directory = "../qgis/sample_results/v1"
    outputs_v1_annual_tifs_directory = "../qgis/sample_results/v1_annual"
    outputs_v1_annual_plots_directory = os.path.join(model_results_v1_plot_directory, "Cumulative Annual Plots")

    model_results_dir = model_results_v1_dir
    model_results_picklefile = model_results_v1_picklefile
    model_results_plot_directory = model_results_v1_plot_directory
    output_tifs_directory = outputs_v1_tifs_directory
    outputs_annual_tifs_directory = outputs_v1_annual_tifs_directory
    outputs_annual_plots_directory = outputs_v1_annual_plots_directory
elif DATA_VERSION == 2:
    model_results_v2_dir = "../data/v2"

    model_results_v2_plot_directory = "../plots/v2"
    outputs_v2_tifs_directory = "../qgis/sample_results/v2"
    outputs_v2_annual_tifs_directory = "../qgis/sample_results/v2_annual"
    outputs_v2_annual_plots_directory = os.path.join(model_results_v2_plot_directory, "Cumulative Annual Maps")
    model_results_v2_picklefile = os.path.join(model_results_v2_dir, "v2_1979-2020.pickle")

    model_results_dir = model_results_v2_dir
    model_results_picklefile = model_results_v2_picklefile
    model_results_plot_directory = model_results_v2_plot_directory
    output_tifs_directory = outputs_v2_tifs_directory
    outputs_annual_tifs_directory = outputs_v2_annual_tifs_directory
    outputs_annual_plots_directory = outputs_v2_annual_plots_directory
elif DATA_VERSION == 2.5:
    # For version 2.5 of the data, with coded filtering thresholds, use this as the default.
    default_melt_code_threshold = 4
    model_results_v2_5_dir = "../data/v2.5"

    model_results_v2_5_picklefile = os.path.join(model_results_v2_5_dir, "v2.5_1979-2021.pickle")
    model_results_v2_5_plot_directory = "../plots/v2.5"

    outputs_v2_5_tifs_directory = "../qgis/sample_results/v2.5"


    outputs_v2_5_annual_tifs_directory = "../data/v2.5/annual_sums"
    outputs_v2_5_annual_plots_directory = os.path.join(model_results_v2_5_plot_directory, "annual_maps")

    v2_5_mean_climatology_tif = os.path.join(model_results_v2_5_dir, "mean_climatologies/1989_2020_mean_climatology.tif")
    v2_5_std_climatology_tif =  os.path.join(model_results_v2_5_dir, "mean_climatologies/1989_2020_std_climatology.tif")

    model_results_dir = model_results_v2_5_dir
    model_results_picklefile = model_results_v2_5_picklefile
    model_results_plot_directory = model_results_v2_5_plot_directory
    output_tifs_directory = outputs_v2_5_tifs_directory
    outputs_annual_tifs_directory = outputs_v2_5_annual_tifs_directory
    outputs_annual_plots_directory = outputs_v2_5_annual_plots_directory
    mean_climatology_geotiff = v2_5_mean_climatology_tif
    std_climatology_geotiff = v2_5_std_climatology_tif
    aws_validation_plots_directory = os.path.join(model_results_v2_5_plot_directory, "aws_validation")
    gap_fill_data_folder = os.path.join(model_results_v2_5_dir, "gap_fill_data")
    daily_melt_averages_picklefile = os.path.join(gap_fill_data_folder, "daily_melt_pixel_averages.pickle")
    daily_cumulative_melt_averages_picklefile = os.path.join(gap_fill_data_folder, "daily_cumulative_melt_averages.pickle")
    gap_filled_melt_picklefile = os.path.join(model_results_v2_5_dir, "v2.5_1979_2021_gap_filled.pickle")
    # The percentiles (10,25,50,75,90) saved in a pandas dataframe.
    baseline_percentiles_csv = os.path.join(model_results_v2_5_dir, "baseline_percentiles_1989-2020.csv")
    # Melt data over each region, each day.
    daily_melt_csv = os.path.join(model_results_v2_5_dir, "daily_melt_totals.csv")

    climatology_plots_directory = os.path.join(model_results_v2_5_plot_directory, "climatology")

elif DATA_VERSION == 3:
    model_results_v3_dir = "../data/"
    model_results_v3_picklefile = os.path.join(model_results_v3_dir, "v3_1979-present_raw.pickle")
    model_results_v3_picklefile_gap_filled = os.path.join(model_results_v3_dir, "v3_1979-present_gap_filled.pickle")

    model_results_dir = os.path.join(model_results_v3_dir, "daily_melt_bin_files")
    model_results_picklefile = model_results_v3_picklefile
    model_results_plot_directory = "../plots/"
    # output_tifs_directory = os.path.join(model_results_v3_dir, "sample_results")
    outputs_annual_tifs_directory = os.path.join(model_results_v3_dir, "annual_sum_geotifs")
    outputs_annual_plots_directory = os.path.join(model_results_plot_directory, "annual_maps_sum")
    mean_climatology_geotiff = os.path.join(model_results_v3_dir, "mean_climatology", "1990_2020_mean_climatology.tif")
    std_climatology_geotiff = os.path.join(model_results_v3_dir, "mean_climatology", "1990_2020_std_climatology.tif")
    gap_fill_data_folder = os.path.join(model_results_v3_dir, "gap_fill_data")
    daily_melt_averages_picklefile = os.path.join(gap_fill_data_folder, "daily_melt_pixel_averages.pickle")
    daily_cumulative_melt_averages_picklefile = os.path.join(gap_fill_data_folder, "daily_cumulative_melt_averages.pickle")
    aws_validation_plots_directory = os.path.join(model_results_plot_directory, "aws_validation")
    # The percentiles (10,25,50,75,90) saved in a pandas dataframe.
    baseline_percentiles_csv = os.path.join(model_results_v3_dir, "baseline_percentiles_1990-2020.csv")
    daily_melt_csv = os.path.join(model_results_v3_dir, "daily_melt_totals.csv")
    gap_filled_melt_picklefile = model_results_v3_picklefile_gap_filled
    climatology_plots_directory = os.path.join(model_results_plot_directory, "annual_line_plots")
    threshold_file_dir = os.path.join(model_results_v3_dir, "thresholds")
    daily_melt_plots_dir = os.path.join(model_results_plot_directory, "daily_maps")
    daily_plots_gathered_dir = os.path.join(model_results_plot_directory, "daily_plots_gathered")
