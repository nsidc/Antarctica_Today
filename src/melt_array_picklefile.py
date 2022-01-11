#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:03:31 2021

create_melt_array_picklefile.py --  Take a list of .bin (or .tif) daily melt
file, generates a picklefile containing an MxNxT numpy array *and* a dictionary
of (datetime:index) pairs for indexing into that array. Save both these files
in a 2-length tuple object (MxNxT array, {datetime:index} dict) in a picklefile
for quick and easy reading.

@author: mmacferrin
"""
from osgeo import gdal
import numpy
import datetime
import pickle
import os
import re

from map_filedata import ice_mask_tif

from tb_file_data import model_results_dir, \
                         model_results_picklefile, \
                         recurse_directory, \
                         gap_filled_melt_picklefile


from read_NSIDC_bin_file import read_NSIDC_bin_file
from progress_bar import ProgressBar

# from ssmi_bin_to_gtif import output_gtif

def get_ice_mask_array(ice_tif = ice_mask_tif):
    """Read the ice mask tif, return the array."""
    ice_mask_ds = gdal.Open(ice_tif, gdal.GA_ReadOnly)
    ice_mask_array = ice_mask_ds.GetRasterBand(1).ReadAsArray()
    return numpy.array(ice_mask_array, dtype=bool)

def find_largest_melt_days_in_an_interval(start_date, end_date, top_n = None, array=None, dt_dict=None):
    """Given a start & end date, find the highest melt day in that interval.

    Useful for finding erroneous days.
    """
    if (array is None) and (dt_dict is None):
        model_array, datetimes_dict = read_model_array_picklefile()
    else:
        model_array, datetimes_dict = array, dt_dict

    melt_pixels = numpy.sum((model_array==2), axis=(0,1))
    assert len(melt_pixels) == len(datetimes_dict)

    # Convert to numpy datetime types, for array comparison below.
    start_date = numpy.datetime64(start_date)
    end_date   = numpy.datetime64(end_date)

    datetimes = list(datetimes_dict.keys())
    datetimes.sort()
    datetimes = numpy.array(datetimes, dtype=numpy.datetime64)
    datetimes_mask = (start_date <= datetimes) & (datetimes <= end_date)

    # datetimes_mask_indices = numpy.where(datetimes_mask)[0]

    datetimes_in_interval = datetimes[datetimes_mask]
    if top_n:
        max_interval_index = numpy.argsort(melt_pixels[datetimes_mask])[(-top_n):][::-1]
    else:
        max_interval_index = numpy.argsort(melt_pixels[datetimes_mask])[-1]

    top_melt = melt_pixels[datetimes_mask][max_interval_index]*(25**2)
    top_dts  = datetimes_in_interval[max_interval_index]

    if top_n:
        print(top_melt)
        print([dt.astype(datetime.datetime).strftime("%Y-%m-%d") for dt in top_dts])
    else:
        print("{0} km2 in {1}".format(top_melt, top_dts.astype(datetime.datetime).strftime("%Y-%m-%d")))

def get_array_from_model_files(file_dir = model_results_dir, verbose=True):
    """Take the individual .bin arrays for each day, and turn it into a M x N x T shaped numpy array."""
    file_list = recurse_directory(file_dir)

    first_file_data = read_NSIDC_bin_file(file_list[0], return_type=int)
    # print(first_file_data.shape)
    # print(first_file_data)
    # print(numpy.unique(first_file_data)) # Values are -1, 0, 1, 2... look from
    # Tom what each of those values actually means.

    # 3D array, Y x X x T
    array_shape = first_file_data.shape + (len(file_list),)
    data_array = numpy.empty(array_shape, dtype=first_file_data.dtype)

    if verbose:
        print("Retrieving melt data from {0} binary (.bin) files.".format(len(file_list)))

    for i,fname in enumerate(file_list):
        if verbose:
            ProgressBar(i+1,len(file_list), length=50, suffix="{0} of {1}".format(i+1, len(file_list)))
        data_array[:,:,i] = read_NSIDC_bin_file(fname, return_type=int)

    return data_array

def save_model_array_picklefile(file_dir = model_results_dir,
                                picklefile = model_results_picklefile):
    """Save the data array *and* the dictionary of datetimes in a picklefile, as a tuple.

    (TxMxN data array, T-length date-->index dictionary)
    """
    data_array = get_array_from_model_files(file_dir = file_dir)
    datetime_dict = get_datetimes_from_file_list(return_as_dict=True)

    f = open(picklefile, 'wb')
    pickle.dump((data_array, datetime_dict), f)
    f.close()

    print(picklefile, "written.")
    return data_array, datetime_dict

def resample_melt_codes_in_array(array, melt_code_threshold):
    """Decode melt to filter out false melt values and give everything a 1 or 2.

    In v2.5 of the data, melt is not just indicated by a "2" value, it has a
    code 2-8, and we've chosen a threshold (currently defaulted to 4) to determine
    if melt actually happened or not. Call this function to change the array
    so that all codes 2 thru melt_code_threshold are classified as "melt" (value 2),
    and all other codes >melt_code_threshold are classified as "no melt" (value 1).
    """
    array[(array > 2) & (array <= melt_code_threshold)] = 2
    array[(array > melt_code_threshold)] = 1
    return array

def read_model_array_picklefile(picklefile = model_results_picklefile,
                                fill_pole_hole = False,
                                filter_out_error_swaths = True,
                                resample_melt_codes = False,
                                resample_melt_code_threshold = 4,
                                verbose=True):
    """Read the model array picklefile.

    Returns a 2-value tuple:
        - MxNxT array of (-1,0,1,2) values,
        - dictionary of {datetimes:index} indices


    "resample_melt_codes" and "resample_melt_code_threshold" are relicts of the v2.5 data,
    and irrelevant to the v3 data.
    Just keep "resample_melt_codes" to False when running with v3 code.
    """
    if verbose:
        print("Reading", os.path.split(picklefile)[-1] + "...", end="")
    f = open(picklefile, 'rb')
    model_array, datetime_dict = pickle.load(f)
    f.close()
    if verbose:
        print("Done.")

    if fill_pole_hole:
        # Fill the pole hole (any missing values) with "no melt" (1)
        row_midpoint = int(model_array.shape[0] / 2) + 8
        col_midpoint = int(model_array.shape[1] / 2)
        row_slice = slice(row_midpoint-23,row_midpoint+23,1)
        col_slice = slice(col_midpoint-23,col_midpoint+23,1)
        model_array[row_slice,col_slice,:][model_array[row_slice,col_slice,:] == 0] = 1

    if resample_melt_codes:
        model_array = resample_melt_codes_in_array(model_array, resample_melt_code_threshold)

    if filter_out_error_swaths:
        model_array = _filter_out_erroneous_swaths(model_array, datetime_dict)

    return model_array, datetime_dict

def get_datetimes_from_file_list(file_dir = model_results_dir,
                                 return_as_dict = False):
    """From the list of file names, which includes an 8-digit YYYYMMDD string, create a list of datetime.datetime objects.

    If "return_as_dict", return a
    dictionary with the datetime objects as keys and the numeric indices into
    the data array as value (see get_array_from_model_files() for details).
    """
    # From tb_file_data.recurse_director() --- finds all *.bin files in the
    # latest-version data directory except in the "thresholds" sub-directory.
    file_list = recurse_directory(file_dir)

    dt_list = [None] * len(file_list)
    for i,fpath in enumerate(file_list):
        search_result = re.search("(?<=_)\d{8}(?=_)", os.path.split(fpath)[-1])
        if search_result is None:
            raise ValueError("Unrecognized file", fpath + ", cannot extract date from file name.")

        datestr = search_result.group()
        dt_list[i] = datetime.datetime(int(datestr[0:4]),
                                       int(datestr[4:6]),
                                       int(datestr[6:8]))

    if return_as_dict:
        ret_dict = {}
        for i,dt in enumerate(dt_list):
            ret_dict[dt] = i
        return ret_dict
    else:
        return dt_list

def _filter_out_erroneous_swaths(model_array, datetimes_dict):
    """Nullify particular false-positive satellite swaths in the data."""
    try:
        # 1985-02-19
        array_slice = model_array[135:175, 210:250, datetimes_dict[datetime.datetime(1985,2,19)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 1985-03-05
        array_slice = model_array[239:260, 180:202, datetimes_dict[datetime.datetime(1985,3,5)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 1985-04-16
        array_slice = model_array[ 90:106, 130:160, datetimes_dict[datetime.datetime(1985,4,16)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 1985-04-22
        array_slice = model_array[120:155, 210:225, datetimes_dict[datetime.datetime(1985,4,22)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 1985-06-23
        array_slice = model_array[140:165, 220:250, datetimes_dict[datetime.datetime(1985,6,23)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 1986-06-06
        array_slice = model_array[210:230, 150:165, datetimes_dict[datetime.datetime(1986,6,6)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 1986-08-31 # Two swaths
        array_slice = model_array[ 80:100, 155:185, datetimes_dict[datetime.datetime(1986,8,31)]]
        array_slice[array_slice == 2] = 0
        array_slice = model_array[230:255, 180:240, datetimes_dict[datetime.datetime(1986,8,31)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    try:
        # 2015-08-06
        array_slice = model_array[100:200,  50:160, datetimes_dict[datetime.datetime(2015,8,6)]]
        array_slice[array_slice == 2] = 0
    except KeyError:
        pass

    return model_array

def read_gap_filled_melt_picklefile(picklefile=gap_filled_melt_picklefile,
                                    verbose=True):
    """Read the gap-filled picklefile, return to user."""
    if verbose:
        print("Reading", picklefile)

    f = open(picklefile,"rb")
    array, dt_dict = pickle.load(f)
    f.close()

    return array, dt_dict


if __name__ == "__main__":
    # Let's save the v2.5 data from Tom's stuff.
    array, dt_dict = save_model_array_picklefile()
