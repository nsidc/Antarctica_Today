#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 08:51:25 2021

@author: mmacferrin
"""
import datetime
import os
import argparse

import melt_array_picklefile
import compute_mean_climatology
import tb_file_data
import generate_antarctica_today_map
import plot_daily_melt_and_climatology
import generate_gap_filled_melt_picklefile


def preprocessing_main():
    """When we get new data (or new versions of the data), do all the things to get it ingested.

    1) Read all the .bin files and put them into the array picklefile
    2) Compute climatology and daily-melt over the baseline period.
    3) Recompute annual melt and annual anomaly geo-tiffs.
    """
    # 1) Read all the .bin files and put them into the array picklefile
    melt_array_picklefile.save_model_array_picklefile()

    # 2) Compute climatology and daily-melt over the baseline period.
    compute_mean_climatology.save_climatologies_as_CSV(gap_filled=False)
    compute_mean_climatology.save_daily_melt_numbers_to_csv(gap_filled=False)
    compute_mean_climatology.create_baseline_climatology_tif(gap_filled=False)

    # 3) Recompute annual melt and annual anomaly geo-tiffs.
    compute_mean_climatology.create_annual_melt_sum_tif(year="all", gap_filled=False)

    compute_mean_climatology.compute_daily_climatology_pixel_averages()
    compute_mean_climatology.compute_daily_sum_pixel_averages()

    # 4) Compute the gap-filled daily melt values (substituting averages)
    generate_gap_filled_melt_picklefile.save_gap_filled_picklefile()

    # 3) Recompute annual melt and annual anomaly geo-tiffs.
    compute_mean_climatology.create_annual_melt_sum_tif(year="all", gap_filled=True)

    # 5) Re-do the mean climatologies with the gap-filled averages.
    compute_mean_climatology.save_climatologies_as_CSV(gap_filled=True)
    compute_mean_climatology.save_daily_melt_numbers_to_csv(gap_filled=True)
    compute_mean_climatology.create_baseline_climatology_tif(gap_filled=True)

    # 6) Loop through all the years. Any "blank" years will simply be ignored.
    for year in range(1979,datetime.datetime.today().year+1):
        compute_mean_climatology.create_annual_melt_anomaly_tif(year, gap_filled=True)

def generate_all_plots_and_maps_main():
    """After all the preprocessing, re-generate all the plots and maps.

    4) Re-run the climatology & daily-melt plots for each year.
    5) Generate new annual maps.
    """
    # 4) Re-run the climatology & daily-melt plots for each year. Years with no data will be skipped (useful if we accidentally go 1 year too far).
    for year in range(1979,datetime.datetime.today().year+1):
        for region in range(0,7+1):

            fname = os.path.join(tb_file_data.climatology_plots_directory, "R{0}_{1}-{2}.png".format(region, year, year+1))

            plot_daily_melt_and_climatology.plot_current_year_melt_over_baseline_stats(\
                                datetime.datetime(year=year+1, month=4, day=30),
                                region_num=region,
                                outfile = fname)

    # 5) Get a quick status check on the dates coverage.
    plot_daily_melt_and_climatology.simple_plot_date_check()

    # 6) Generate new annual maps.
    mapper = generate_antarctica_today_map.AT_map_generator(gap_filled=True)
    mapper.generate_annual_melt_map(dpi=300,
                                    year="all",
                                    reset_picklefile=True)

    mapper.generate_anomaly_melt_map(dpi=300,
                                     year="all",
                                     reset_picklefile=True)

    # Create the latest anomaly map for a partial year (useful during the melt season).
    mapper.generate_latest_partial_anomaly_melt_map(dpi=300)


# def generate_all_plots_for_a_given_year_main(year=2021):
#     """After all the pre-processing, generate the plots and maps (in both PNG and SVG).

#     NOTE: I never finished all this"""
#     for region in range(0,7+1):

#         fname = os.path.join(tb_file_data.climatology_plots_directory, "R{0}_{1}-{2}.png".format(region, year, year+1))
#         plot_daily_melt_and_climatology.plot_current_year_melt_over_baseline_stats(\
#                             datetime.datetime(year=year+1, month=4, day=30),
#                             region_num=region,
#                             outfile = fname)

#         fname_svg = os.path.join(tb_file_data.climatology_plots_directory, "R{0}_{1}-{2}.svg".format(region, year, year+1))
#         plot_daily_melt_and_climatology.plot_current_year_melt_over_baseline_stats(\
#                             datetime.datetime(year=year+1, month=4, day=30),
#                             region_num=region,
#                             outfile = fname_svg)

#     # TODO: Finish if you want.

def read_and_parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=
                                     'Generate preprocessing or processing steps for Antarctica Today data.')
    parser.add_argument('--preprocess',
                        help="Generate all pre-processed data.",
                        default=False,
                        action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = read_and_parse_args()

    if args.preprocess:
        preprocessing_main()
    else:
        generate_all_plots_and_maps_main()
