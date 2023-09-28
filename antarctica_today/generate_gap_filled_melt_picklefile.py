"""Code for generating the gap-filled melt array picklefile.
Putting this inside of melt_array_picklefile.py caused circular-dependency issues since it uses
a function from compute_mean_climatology, which uses functions from melt_array_picklefile. So,
we put this here instead."""


import datetime
import pickle

import numpy
from compute_mean_climatology import read_daily_melt_averages_picklefile
from melt_array_picklefile import read_model_array_picklefile
from tb_file_data import gap_filled_melt_picklefile


def save_gap_filled_picklefile(picklefile=gap_filled_melt_picklefile, verbose=True):
    """Write the picklefile."""
    array, datetimes_dict = fill_melt_array_with_interpolations(verbose=verbose)

    f = open(picklefile, "wb")
    pickle.dump((array, datetimes_dict), f)
    f.close()
    if verbose:
        print(picklefile, "written.")


def fill_melt_array_with_interpolations(array=None, datetimes_dict=None, verbose=True):
    """Take the mean melt array, fill it with interpolations from the mean climatology.

    The array and datetimes_dict can be sent as parameters if they've already been read.
    This avoids unnecessary file redundancy.

    NOTE: This returns an array with different codes, due to the floating-point values used.

    Input: integer array, (-1,0,1,2) == (off_ice, no-data, no-melt, melt)
    Output: float array, (-1,0..1) == (off_ice, fractional probability of melt on that day. 0 == no melt measured. 1 = melt measured, 0.x = missing data but something in-between.)
    """
    # Get the dates and
    if array == None or datetimes_dict == None:
        array, datetimes_dict = read_model_array_picklefile(
            resample_melt_codes=True, verbose=verbose
        )

    avg_array, avg_dt_dict = read_daily_melt_averages_picklefile(
        build_picklefile_if_not_present=True
    )
    # Iterate through and build an array of datetimes and gap-filled arrays.
    # Skip the days that aren't in the melt season.
    # Start at the very beginning of the time series.
    # NOTE: There is surely a more efficient way of doing this. This is just easy,
    # so if it's "fast enough", I'll use it.
    dt_list = list(datetimes_dict.keys())
    first_day_dt = numpy.min(dt_list)
    last_day_dt = numpy.max(dt_list)

    gap_filled_dt_list = [
        first_day_dt + datetime.timedelta(days=d)
        for d in range((last_day_dt - first_day_dt).days + 1)
    ]
    # Filter to melt-season only:
    gap_filled_dt_list = [
        dt for dt in gap_filled_dt_list if (dt.month, dt.day) in avg_dt_dict
    ]

    assert gap_filled_dt_list[-1] == last_day_dt

    gap_filled_dt_dict = dict([(dt, i) for i, dt in enumerate(gap_filled_dt_list)])

    # print("original array:")
    # print("\t",dt_list[0], dt_list[-1], len(dt_list))
    # print("new array:")
    # print("\t",gap_filled_dt_list[0], gap_filled_dt_list[-1], len(gap_filled_dt_list))

    gap_filled_array = numpy.empty(
        (array.shape[0], array.shape[1], len(gap_filled_dt_list)), dtype=numpy.float
    )

    for gap_i, dt in enumerate(gap_filled_dt_list):
        # Case 1: this date is in the array. fill in the "0" (nodata) values
        # print(dt.strftime("%Y-%m-%d"), "==================")
        if dt in datetimes_dict:
            day_slice = array[:, :, datetimes_dict[dt]]
            day_slice_missing_mask = day_slice == 0
            # Convert codes from (0,1,2) (no-data, no-melt, melt) to (0,1) (no-melt, melt, or a probability thereof)
            # print(0,numpy.count_nonzero(day_slice==0))
            # print(1,numpy.count_nonzero(day_slice==1))
            # print(2,numpy.count_nonzero(day_slice==2))
            day_slice[day_slice == 1] = 0
            day_slice[day_slice == 2] = 1

            # print("\t", numpy.count_nonzero(day_slice_missing_mask), "missing.")
            # print("\t", "Before filling mean melt:", numpy.sum(day_slice[day_slice != -1]))
            gap_filled_array[:, :, gap_i] = day_slice
            gap_filled_array[:, :, gap_i][day_slice_missing_mask] = avg_array[
                :, :, avg_dt_dict[(dt.month, dt.day)]
            ][day_slice_missing_mask]
            # print("\t", "After filling mean melt:", numpy.sum(day_slice[day_slice != -1]))

        # Case 2: this melt day is omitted from the array. Just substitute the mean values for that day-of-year.
        else:
            day_slice = avg_array[:, :, avg_dt_dict[(dt.month, dt.day)]]
            gap_filled_array[:, :, gap_i] = day_slice
            # print("\t", "Day missing, fill with mean. Average:", numpy.sum(day_slice[day_slice!=-1]))

    return gap_filled_array, gap_filled_dt_dict
