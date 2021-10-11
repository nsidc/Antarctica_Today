# -*- coding: utf-8 -*-

"""Scripts for importing new NSIDC brightness-temperature data into the Antarctic melt database.

Created by: mmacferrin
2021.04.08
"""
# from read_NSIDC_bin_file import read_NSIDC_bin_file
import os
import tb_file_data
import re
import numpy
import datetime
# import melt_array_picklefile
# import nsidc_download_Tb_data
import generate_daily_melt_file
import pickle
import compute_mean_climatology
import generate_gap_filled_melt_picklefile

# THIS IS HARD-CODED FOR NOW. TODO: Change it to be more dynamice and adaptable.
melt_bin_dir = "../data/v2.5/antarctica_melt_S3B_2021-2021-20210411"


def get_list_of_NSIDC_bin_files_to_import(datadir=tb_file_data.NSIDC_0080_file_dir,
                                          hemisphere="S",
                                          frequencies=[19,37],
                                          polarization='v',
                                          target_extension=".bin"):
    """Read the directory and import a list of NSIDC .bin Tb files to open & import."""
    # Get a list of all files in the directory with that extension.
    file_list_all = [f for f in os.listdir(datadir) if os.path.splitext(f)[1].lower() == target_extension.strip().lower()]

    # print(len(file_list_all), "total files.")

    # Filter out only the files we want.
    hemisphere_lower = hemisphere.lower()
    polarization_lower = polarization.lower()
    search_template = r'^tb_f\d{2}_\d{8}_nrt_' + hemisphere_lower + \
                      "(" + "|".join(["({0:02d})".format(freq) for freq in frequencies]) + ")" + \
                      polarization_lower + '.bin$'

    # print(search_template)
    # Create a compiled regular-expression search object
    pattern = re.compile(search_template)
    # Keep only the file names that match the search pattern.
    file_list_filtered = [fname for fname in file_list_all if pattern.search(fname)]
    file_list_filtered.sort()

    file_paths_list = [os.path.join(datadir, fname) for fname in file_list_filtered]

    # print(len(file_paths_list), "filtered files.")

    return file_paths_list

# def select_only_new_files(file_path_list):
#     """From a list of files, return a list of only "new" files that need to be read.

#     Remove file paths for dates that are already in the current dataset.
#     """
#     # TODO: Finish

def update_everything_to_latest_date(overwrite = True):
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
        dt = datetime.datetime(year=int(datestr[0:4]), month=int(datestr[4:6]), day=int(datestr[6:8]))
        if dt > latest_dt:
            latest_dt = dt

    # start_time_str = (latest_dt_in_array + datetime.timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    # Download all Tb files (19 & 37 GHz vertical), starting with the day
    # after the last date in the present array.
    # tb_file_list = nsidc_download_Tb_data.download_new_files(time_start=start_time_str)
    # # Ignore the .xml files, only get a list of the .bin files we downloaded.
    # tb_file_list = [fname for fname in tb_file_list if os.path.splitext(fname)[-1].lower() == ".bin"]
    tb_file_list = [os.path.join("../Tb/nsidc-0080", fname) for fname in os.listdir("../Tb/nsidc-0080") if os.path.splitext(fname)[-1] == ".bin"]

    # Define "today" as today at midnight.
    dt_today = datetime.datetime.today()
    dt_today = datetime.datetime(year=dt_today.year, month=dt_today.month, day=dt_today.day)

    daily_melt_arrays = []
    daily_dts = []
    for day_delta in range(1,((dt_today - latest_dt).days + 1)):
        dt = latest_dt + datetime.timedelta(days=day_delta)
        fnames_19h = [fname for fname in tb_file_list if \
                      (fname.find(dt.strftime("%Y%m%d")) > -1 and fname.find("s19h.bin") > -1)]
        fnames_37h = [fname for fname in tb_file_list if \
                      (fname.find(dt.strftime("%Y%m%d")) > -1 and fname.find("s37h.bin") > -1)]
        # Make sure there's just one of each file. If not, figure out what's going on here.
        try:
            assert len(fnames_19h) == 1
            assert len(fnames_37h) == 1
        except AssertionError:
            if len(fnames_19h) == 0 or len(fnames_37h) == 0:
                continue
            # print(dt)
            # print(dt.strftime("%Y%m%d"))
            # print(fnames_19h, fnames_37h)
            # foobar

        # Generate the name of the output .bin file.
        melt_bin_fname = os.path.join(melt_bin_dir, dt.strftime("antarctica_melt_%Y%m%d_S3B_20210411.bin"))

        # Create new .bin files for each new melt day.
        melt_array = generate_daily_melt_file.create_daily_melt_file(fnames_19h[0],
                                                                     fnames_37h[0],
                                                                     melt_bin_fname)

        # Add a third dimension to aid in concatenating
        melt_array.shape = (melt_array.shape[0], melt_array.shape[1], 1)

        daily_dts.append(dt)
        daily_melt_arrays.append(melt_array)

    new_melt_array = numpy.concatenate(daily_melt_arrays, axis=2)

    # Concatenate the new melt array with the old one.
    melt_array_updated = numpy.concatenate((melt_array, new_melt_array), axis=2)
    # Add all the new datetimes to the dictionary
    dt_dict = dict()
    for i,dt in enumerate(daily_dts):
        dt_dict[dt] = i+1

    if overwrite:
        f = open(tb_file_data.model_results_picklefile, 'wb')
        pickle.dump((melt_array_updated, dt_dict), f)
        f.close()

        print(tb_file_data.model_results_picklefile, "written.")

    # Interpolate the gaps.
    generate_gap_filled_melt_picklefile.save_gap_filled_picklefile()

    # Now that we have the arrays updated, let's re-compute the daily melt sums.
    compute_mean_climatology.save_daily_melt_numbers_to_csv(gap_filled=False)
    compute_mean_climatology.save_daily_melt_numbers_to_csv(gap_filled=True)

    return melt_array_updated, dt_dict


# TODO: Made function to update/append the daily-melt CSV file.

if __name__ == "__main__":
    update_everything_to_latest_date()
