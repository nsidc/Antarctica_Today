"""Scripts for importing new NSIDC brightness-temperature data into the Antarctic melt database.

Created by: mmacferrin
2021.04.08
"""
import datetime
import os
import pickle
import re
import shutil

import dateutil
import numpy

from antarctica_today import (
    generate_daily_melt_file,
    map_filedata,
    tb_file_data,
)
from antarctica_today.compute_mean_climatology import save_daily_melt_numbers_to_csv
from antarctica_today.constants.paths import DATA_TB_DIR
from antarctica_today.generate_antarctica_today_map import AT_map_generator
from antarctica_today.generate_gap_filled_melt_picklefile import (
    save_gap_filled_picklefile,
)
from antarctica_today.melt_array_picklefile import read_model_array_picklefile
from antarctica_today.nsidc_download_Tb_data import download_new_files
from antarctica_today.plot_daily_melt_and_climatology import (
    plot_current_year_melt_over_baseline_stats,
)
from antarctica_today.read_NSIDC_bin_file import read_NSIDC_bin_file


def get_list_of_NSIDC_bin_files_to_import(
    datadir=tb_file_data.NSIDC_0080_file_dir,
    hemisphere="S",
    frequencies=[19, 37],
    polarization="v",
    target_extension=".bin",
):
    """Read the directory and import a list of NSIDC .bin Tb files to open & import."""
    # Get a list of all files in the directory with that extension.
    file_list_all = [
        f
        for f in os.listdir(datadir)
        if os.path.splitext(f)[1].lower() == target_extension.strip().lower()
    ]

    # print(len(file_list_all), "total files.")

    # Filter out only the files we want.
    hemisphere_lower = hemisphere.lower()
    polarization_lower = polarization.lower()
    search_template = (
        r"^tb_f\d{2}_\d{8}_nrt_"
        + hemisphere_lower
        + "("
        + "|".join(["({0:02d})".format(freq) for freq in frequencies])
        + ")"
        + polarization_lower
        + ".bin$"
    )

    # print(search_template)
    # Create a compiled regular-expression search object
    pattern = re.compile(search_template)
    # Keep only the file names that match the search pattern.
    file_list_filtered = [fname for fname in file_list_all if pattern.search(fname)]
    file_list_filtered.sort()

    file_paths_list = [os.path.join(datadir, fname) for fname in file_list_filtered]

    return file_paths_list


def update_everything_to_latest_date(
    overwrite=True,
    melt_bin_dir=tb_file_data.model_results_dir,
    copy_to_gathered_dir=True,
    melt_season_only=True,
    melt_season_start_mmdd=[10, 1],
    melt_season_end_mmdd=[4, 30],
    date_today=None,
):
    """Using today's date, do everything to update with the newest data.

    1) Download the NSIDC Tb files that are missing.
    2) Generate daily melt files from those Tb files.
    3) Update the database to include those new daily melt files, overwrite it with current data included.
    4) Update the Daily Melt CSV file with summary data.
    """
    existing_bin_files_list = os.listdir(tb_file_data.model_results_dir)
    latest_dt = datetime.datetime(year=1900, month=1, day=1)
    for fname in existing_bin_files_list:
        datestr = re.search("(?<=antarctica_melt_)\d{8}(?=_S3B)", fname).group()
        dt = datetime.datetime(
            year=int(datestr[0:4]), month=int(datestr[4:6]), day=int(datestr[6:8])
        )
        if dt > latest_dt:
            latest_dt = dt

    start_time_str = (latest_dt + datetime.timedelta(days=1)).strftime(
        "%Y-%m-%dT00:00:00Z"
    )

    # Download all Tb files (19 & 37 GHz vertical), starting with the day
    # after the last date in the present array.
    # TODO: We're overwriting this variable a couple lines below; do we need it?
    tb_file_list = download_new_files(time_start=start_time_str)
    # # Ignore the .xml files, only get a list of the .bin files we downloaded.
    # tb_file_list = [fname for fname in tb_file_list if os.path.splitext(fname)[-1].lower() == ".bin"]
    tb_0080_dir = DATA_TB_DIR / "nsidc-0080"
    tb_file_list = [fp for fp in tb_0080_dir.iterdir() if fp.suffix == ".bin"]

    # Define "today" as today at midnight.
    if date_today is None:
        dt_today = datetime.datetime.today()
        dt_today = datetime.datetime(
            year=dt_today.year, month=dt_today.month, day=dt_today.day
        )
    else:
        if type(date_today) in (datetime.datetime, datetime.date):
            dt_today = date_today
        else:
            dt_today = dateutil.parser.parse(date_today)

    # Collect all the new Tb .bin files and create a daily melt .bin file from each.
    for day_delta in range(1, ((dt_today - latest_dt).days + 1)):
        dt = latest_dt + datetime.timedelta(days=day_delta)

        if melt_season_only:
            dt_mmdd = [dt.month, dt.day]
            if (
                (melt_season_start_mmdd > melt_season_end_mmdd)
                and (melt_season_end_mmdd < dt_mmdd < melt_season_start_mmdd)
            ) or (
                (melt_season_start_mmdd <= melt_season_end_mmdd)
                and (
                    (dt_mmdd < melt_season_start_mmdd)
                    or (dt_mmdd > melt_season_end_mmdd)
                )
            ):
                continue

        fnames_37h = [
            fname
            for fname in tb_file_list
            if (
                fname.find(dt.strftime("%Y%m%d")) > -1
                and (fname.find("s37h.bin") > -1)
                and (fname.find(".xml") == -1)
            )
        ]
        fnames_19v = [
            fname
            for fname in tb_file_list
            if (
                fname.find(dt.strftime("%Y%m%d")) > -1
                and (fname.find("s19v.bin") > -1)
                and (fname.find(".xml") == -1)
            )
        ]
        fnames_37v = [
            fname
            for fname in tb_file_list
            if (
                fname.find(dt.strftime("%Y%m%d")) > -1
                and (fname.find("s37v.bin") > -1)
                and (fname.find(".xml") == -1)
            )
        ]
        # Make sure there's just one of each file. If not, figure out what's going on here.
        try:
            assert len(fnames_37h) == 1
            assert len(fnames_37v) == 1
            assert len(fnames_19v) == 1
        except AssertionError as e:
            if len(fnames_19v) == 0 or len(fnames_37h) == 0 or len(fnames_37v) == 0:
                continue
            else:
                raise e

        # Generate the name of the output .bin file.
        melt_bin_fname = os.path.join(
            melt_bin_dir,
            dt.strftime(
                "antarctica_melt_%Y%m%d_S3B_{0}.bin".format(dt_today.strftime("%Y%m%d"))
            ),
        )

        threshold_file = generate_daily_melt_file.get_correct_threshold_file(dt)
        if threshold_file is None:
            continue

        # Create new .bin files for each new melt day.
        melt_array = generate_daily_melt_file.create_daily_melt_file(
            fnames_37h[0], fnames_37v[0], fnames_19v[0], threshold_file, melt_bin_fname
        )

        # Add a third dimension to aid in concatenating with the larger melt array.
        melt_array.shape = (melt_array.shape[0], melt_array.shape[1], 1)

    # Now, get a list of all melt arrays that aren't yet in the melt array picklefile.
    melt_bin_files = sorted(os.listdir(melt_bin_dir))
    melt_bin_paths = [os.path.join(melt_bin_dir, fn) for fn in melt_bin_files]
    melt_array, dt_dict = read_model_array_picklefile()
    latest_dt_in_array = max(dt_dict.keys())

    daily_melt_arrays = []
    daily_dts = []

    # For each day, find the .bin file for that day (if it exists) and append it to the list.
    for day_delta in range(1, ((dt_today - latest_dt_in_array).days + 1)):
        dt = latest_dt_in_array + datetime.timedelta(days=day_delta)
        melt_filepath = None
        melt_filename = None
        for i, fn in enumerate(melt_bin_files):
            if fn.find("antarctica_melt_{0}_S3B".format(dt.strftime("%Y%m%d"))) > -1:
                melt_filename = fn
                melt_filepath = melt_bin_paths[i]
                break
        # If this day doesn't have a .bin file associated with it, just skip it.
        if melt_filepath is None:
            continue

        print(melt_filename, "read.")
        daily_melt_array = read_NSIDC_bin_file(
            melt_filepath, element_size=2, return_type=int, signed=True, multiplier=1
        )
        # Add a 3rd (time) dimension to each array to allow concatenating.
        daily_melt_array.shape = list(daily_melt_array.shape) + [1]

        daily_dts.append(dt)
        daily_melt_arrays.append(daily_melt_array)

    if len(daily_melt_arrays) > 0:
        new_melt_array = numpy.concatenate(daily_melt_arrays, axis=2)

        # Concatenate the new melt array with the old one.
        melt_array_updated = numpy.concatenate((melt_array, new_melt_array), axis=2)
        # Add all the new datetimes to the dictionary
        for i, dt in enumerate(daily_dts):
            dt_dict[dt] = melt_array.shape[2] + i

        if overwrite:
            f = open(tb_file_data.model_results_picklefile, "wb")
            pickle.dump((melt_array_updated, dt_dict), f)
            f.close()

        print(tb_file_data.model_results_picklefile, "written.")

    else:
        melt_array_updated = melt_array

    if len(daily_melt_arrays) > 0:
        # Interpolate the gaps.
        save_gap_filled_picklefile()

        # Now that we have the arrays updated, let's re-compute the daily melt sums.
        save_daily_melt_numbers_to_csv(gap_filled=False)
        save_daily_melt_numbers_to_csv(gap_filled=True)

    # Generate the latest current-day maps.
    mapper = AT_map_generator()
    if len(daily_dts) > 0:
        year = generate_daily_melt_file.get_melt_year_of_current_date(daily_dts[-1])
    else:
        year = generate_daily_melt_file.get_melt_year_of_current_date(
            latest_dt_in_array
        )

    latest_date = sorted(dt_dict.keys())[-1]
    date_message = "through " + latest_date.strftime("%d %b %Y").lstrip("0")

    # Plot the latest line plots.
    for region_num in range(0, 7 + 1):
        mapper.generate_annual_melt_map(
            year=year,
            region_number=region_num,
            mmdd_of_year=(latest_date.month, latest_date.day),
            dpi=300,
            message_below_year=date_message,
        )
        mapper.generate_latest_partial_anomaly_melt_map(
            dpi=300,
            region_number=region_num,
            message_below_year=date_message + "\nrelative to 1990-2020 averages",
        )
        mapper.generate_daily_melt_map(
            infile="latest", outfile="auto", region_number=region_num, dpi=300
        )

        outfile = os.path.join(
            tb_file_data.climatology_plots_directory,
            "R{0}_{1}-{2}_{3}_gap_filled.png".format(
                region_num, year, year + 1, latest_date.strftime("%Y.%m.%d")
            ),
        )
        plot_current_year_melt_over_baseline_stats(
            current_date=latest_date,
            region_num=region_num,
            gap_filled=True,
            outfile=outfile,
            verbose=True,
        )

    if copy_to_gathered_dir:
        copy_latest_date_plots_to_date_directory(
            year=generate_daily_melt_file.get_melt_year_of_current_date(dt_today),
            date=latest_date,
        )

    return melt_array_updated, dt_dict


def copy_latest_date_plots_to_date_directory(
    year=generate_daily_melt_file.get_melt_year_of_current_date(
        datetime.datetime.today() - datetime.timedelta(days=1)
    ),
    date=datetime.datetime.today() - datetime.timedelta(days=1),
    dest_parent_dir=tb_file_data.daily_plots_gathered_dir,
    daily_melt_maps_dir=map_filedata.daily_maps_directory,
    sum_maps_dir=map_filedata.annual_maps_directory,
    anomaly_maps_dir=map_filedata.anomaly_maps_directory,
    line_plots_dir=tb_file_data.climatology_plots_directory,
    verbose=True,
):
    """After running the 'update_everything_to_latest_date()' function, use this to gather all the
    latest-date plots into one location. Put it in a sub-directory of the daily_plots_gathered_dir
    with all the latest plots just made."""

    # Get latest file from melt anomaly directory. Also, get the date we're computing up to.
    date_string = date.strftime("%Y.%m.%d")
    search_str = (
        "R[0-7]_{0}-{1}_{2}".format(year, year + 1, date_string) + "(\w)*\.png\Z"
    )
    # Get all the files that match our search string. May be different dates and possibly different regions.

    files_in_anomaly_maps_dir = [
        os.path.join(anomaly_maps_dir, fn)
        for fn in os.listdir(anomaly_maps_dir)
        if re.search(search_str, fn) is not None
    ]
    files_in_sum_maps_dir = [
        os.path.join(sum_maps_dir, fn)
        for fn in os.listdir(sum_maps_dir)
        if re.search(search_str, fn) is not None
    ]
    # The daily maps don't list the melt year, so we use a slightly different search string for those.
    daily_search_str = "R[0-7]_{0}_daily.png".format(date_string)
    files_in_daily_maps_dir = [
        os.path.join(daily_melt_maps_dir, fn)
        for fn in os.listdir(daily_melt_maps_dir)
        if re.search(daily_search_str, fn) is not None
    ]
    files_in_line_plots_dir = [
        os.path.join(line_plots_dir, fn)
        for fn in os.listdir(line_plots_dir)
        if re.search(search_str, fn) is not None
    ]

    files_to_move = (
        files_in_anomaly_maps_dir
        + files_in_sum_maps_dir
        + files_in_daily_maps_dir
        + files_in_line_plots_dir
    )

    # If the sub-directory with this latest date doesn't exist, create it.
    dest_dir_location = os.path.join(dest_parent_dir, date_string)
    if not os.path.exists(dest_dir_location):
        os.mkdir(dest_dir_location)
        if verbose:
            print("Created directory '{0}'.".format(dest_dir_location))

    for fn in files_to_move:
        src = fn
        dst = os.path.join(dest_dir_location, os.path.basename(fn))
        if os.path.exists(dst) and os.path.exists(src):
            os.remove(dst)

        shutil.copyfile(src, dst)

        if verbose:
            print("{0} -> {1}.".format(src, dst))


if __name__ == "__main__":
    update_everything_to_latest_date(
        date_today=datetime.datetime.today().strftime("%Y.%m.%d"),
        copy_to_gathered_dir=True,
    )
    # update_everything_to_latest_date(date_today="2022.04.30")
    # copy_latest_date_plots_to_date_directory()
