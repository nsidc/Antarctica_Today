"""
Written by: Mike MacFerrin, 2023.12.13 (original version)
"""

import argparse
import datetime
import dateutil.parser
import matplotlib.pyplot
import os
import re
import shutil

from antarctica_today import (
    generate_daily_melt_file,
    generate_antarctica_today_map,
    tb_file_data,
    plot_daily_melt_and_climatology,
    map_filedata
)

def generate_maps_and_plots_for_a_date(
    dt: datetime.datetime,
    region="ALL",
    dpi: int=300,
    copy_to_gathered_dir: bool=True,
    ):
    """Generate 3 maps and one line-plot for a given date.
    
    If a region number is given (0-7), produce them just for that region. If 'ALL', produce them for all the regions.
    Plots will be placed in their respective sub-directories under /plots.
    
    If copy_to_gathered_dir, after plots are produced for that day, put them in the "/plots/daily_plots_gathered"
    sub-directory (under a sub-dir matching that date) for easy fetching.
    
    This function does not do any of the processing for a given date nor update the database. For that, use
    the update_data.py script.
    """
    # QA the "region" parameter
    if type(region) is str:
        if region.strip().upper() == "ALL":
            region = "ALL"
        else:
            # Convert to an integer. If it's already an integer (or float) this will work already.
            # If it's something else, it'll error-out usefully (the message to the user will probably be useful).
            region = int(region)

    if region == "ALL":
        regions = list(range(0,8))
    else:
        assert type(region) == int and 0 <= region <= 7
        regions = [region]

    date_message = "through " + dt.strftime("%d %b %Y").lstrip("0")
    year = generate_daily_melt_file.get_melt_year_of_current_date(dt)
    mapper = generate_antarctica_today_map.AT_map_generator()

    for region_num in regions:
        mapper.generate_annual_melt_map(
            year=year,
            region_number=region_num,
            mmdd_of_year=(dt.month, dt.day),
            dpi=dpi,
            message_below_year=date_message
        )

        mapper.generate_anomaly_melt_map(
            year=year,
            mmdd_of_year=(dt.month, dt.day),
            dpi=dpi,
            region_number=region_num,
            message_below_year=date_message + "\nrelative to 1990-2020 average",
        )

        # Find the daily melt .bin file for this particular day, to create the daily melt map for that day.
        daily_dir = tb_file_data.model_results_dir
        matching_binfiles = \
            [os.path.join(daily_dir, fname)
                for fname in os.listdir(daily_dir) if
                fname.find("antarctica_melt_" + dt.strftime("%Y%m%d")) == 0 and os.path.splitext(fname)[1] == ".bin"
            ]
        # We really should only find one melt .bin file for that particular day.
        assert len(matching_binfiles) <= 1
        # Only generate a map if we've found a file for it.
        if len(matching_binfiles) == 1:
            mapper.generate_daily_melt_map(
                infile=matching_binfiles[0], outfile="auto", region_number=region_num, dpi=dpi
            )

        lineplot_outfile = os.path.join(
            tb_file_data.climatology_plots_directory,
            "R{0}_{1}-{2}_{3}_gap_filled.png".format(
                region_num, year, year + 1, dt.strftime("%Y.%m.%d")
            ),
        )

        plot_daily_melt_and_climatology.plot_current_year_melt_over_baseline_stats(
            current_date=dt,
            region_num=region_num,
            gap_filled=True,
            dpi=dpi,
            outfile=lineplot_outfile,
            verbose=True,
        )

        # Close the current plots open in matplotlib. (Keeps them from accumulating.)
        matplotlib.pyplot.close("all")

    if copy_to_gathered_dir:
        """After running the 'update_everything_to_latest_date()' function, use this to gather all the
        latest-date plots into one location. Put it in a sub-directory of the daily_plots_gathered_dir
        with all the latest plots just made."""

        # Get latest file from melt anomaly directory. Also, get the date we're computing up to.
        date_string = dt.strftime("%Y.%m.%d")
        if region == "ALL":
            search_str = "R[0-7]_{0}-{1}_{2}".format(year, year + 1, date_string) + r"(\w)*\.png\Z"
        else:
            search_str = "R{3}_{0}-{1}_{2}".format(year, year + 1, date_string, region) + r"(\w)*\.png\Z"

        # Get all the files that match our search string. May be different dates and possibly different regions.

        # daily_melt_maps_dir = map_filedata.daily_maps_directory,
        # sum_maps_dir = map_filedata.annual_maps_directory,
        # anomaly_maps_dir = map_filedata.anomaly_maps_directory,
        # line_plots_dir = tb_file_data.climatology_plots_directory

        files_in_anomaly_maps_dir = [
            os.path.join(map_filedata.anomaly_maps_directory, fn)
            for fn in os.listdir(map_filedata.anomaly_maps_directory)
            if re.search(search_str, fn) is not None
        ]
        files_in_sum_maps_dir = [
            os.path.join(map_filedata.annual_maps_directory, fn)
            for fn in os.listdir(map_filedata.annual_maps_directory)
            if re.search(search_str, fn) is not None
        ]

        # The daily maps don't list the melt year, so we use a slightly different search string for those.
        daily_search_str = "R[0-7]_{0}_daily.png".format(date_string)
        files_in_daily_maps_dir = [
            os.path.join(map_filedata.daily_maps_directory, fn)
            for fn in os.listdir(map_filedata.daily_maps_directory)
            if re.search(daily_search_str, fn) is not None
        ]
        files_in_line_plots_dir = [
            os.path.join(tb_file_data.climatology_plots_directory, fn)
            for fn in os.listdir(tb_file_data.climatology_plots_directory)
            if re.search(search_str, fn) is not None
        ]

        files_to_move = (
                files_in_anomaly_maps_dir
                + files_in_sum_maps_dir
                + files_in_daily_maps_dir
                + files_in_line_plots_dir
        )

        date_string = dt.strftime("%Y.%m.%d")
        # If the sub-directory with this latest date doesn't exist, create it.
        dest_dir_location = os.path.join(tb_file_data.daily_plots_gathered_dir, date_string)
        if not os.path.exists(dest_dir_location):
            os.mkdir(dest_dir_location)
            print("Created directory '{0}'.".format(dest_dir_location))

        for fn in files_to_move:
            src = fn
            dst = os.path.join(dest_dir_location, os.path.basename(fn))
            if os.path.exists(dst) and os.path.exists(src):
                os.remove(dst)

            shutil.copyfile(src, dst)

            print("{0} -> {1}.".format(src, dst))


def define_and_parse_args():
    parser = argparse.ArgumentParser("Produce all plots for the season on a particular date. Does not download any new"
                                     " data or update the database, just produces the plots and maps.")
    parser.add_argument("DATE", type=str, help="A date in a format readable by the python dateutil.parser module."
                                               "YYYY-MM-DD suffices."
                                               "Date should be within the melt season from 01 Oct thru 30 Apr."
                                               "Behavior is undefined for dates outside that melt-season range.")
    parser.add_argument("-region", "-r", default="ALL",
                        help="Generate for a specific Antarctic region (0-7), or 'ALL'. Default 'ALL' will produce all"
                             "regions 0 (Antarctic continent) and 1-7 (or each sub-region).")
    parser.add_argument("-dpi", "-d", type=int, default=300, help="DPI of output figures (in PNG format)."
                                                                               " Default 300.")
    parser.add_argument("--gather", "--g", default=False, action="store_true",
                        help="Copy all plots produced into a dated sub-dir of the plots/daily_plots_gathered for easy"
                        "fetching after execution.")

    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    this_dt = dateutil.parser.parse(args.DATE)

    generate_maps_and_plots_for_a_date(
        this_dt,
        region=args.region,
        dpi=args.dpi,
        copy_to_gathered_dir=args.gather
    )