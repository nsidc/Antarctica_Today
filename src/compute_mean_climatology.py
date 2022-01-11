# -*- coding: utf-8 -*-

import datetime
import numpy
import os
from osgeo import gdal
import pandas
import pickle

from melt_array_picklefile import read_model_array_picklefile, \
                                  get_ice_mask_array, \
                                  read_gap_filled_melt_picklefile

from tb_file_data import mean_climatology_geotiff, \
                         std_climatology_geotiff, \
                         outputs_annual_tifs_directory, \
                         baseline_percentiles_csv, \
                         daily_melt_csv, \
                         antarctic_regions_tif, \
                         antarctic_regions_dict, \
                         daily_melt_averages_picklefile, \
                         daily_cumulative_melt_averages_picklefile

from write_NSIDC_bin_to_gtif import output_gtif

def compute_daily_climatology_pixel_averages(baseline_start_year = 1990,
                                             melt_start_mmdd=(10,1),
                                             baseline_end_year = 2020,
                                             melt_end_mmdd=(4,30),
                                             output_picklefile=daily_melt_averages_picklefile,
                                             verbose=True):
    """Compute fraction of days in the baseline period in which each give pixel melts.

    Use the baseline period. Calculate, of the days with data (ignoring
    no-data days at any given pixel, what the odds are 0..1 of it having melted
    that day over the baseline period.

    Return an MxNxT array and a dictionary of {(mm,dd):i} indices into the
    array for each (month,day) of the defined melt season.
    Save them to the picklefile listed.
    """
    melt_array, datetimes_dict = read_model_array_picklefile(resample_melt_codes=True)

    # Recode melt array from (-1, 0, 1, 2), to (nan, nan, 0.0, 1.0), and convert to floating-point
    melt_array_nan_filled = numpy.array(melt_array, dtype=numpy.float32)
    melt_array_nan_filled[melt_array_nan_filled == -1.0] = numpy.nan
    melt_array_nan_filled[melt_array_nan_filled ==  0.0] = numpy.nan
    melt_array_nan_filled[melt_array_nan_filled ==  1.0] = 0.0
    melt_array_nan_filled[melt_array_nan_filled ==  2.0] = 1.0

    datetimes_list = list(datetimes_dict.keys())

    # baseline_start_date = datetime.datetime(baseline_start_year, melt_start_mmdd[0], melt_start_mmdd[1])
    # baseline_end_date   = datetime.datetime(baseline_end_year  , melt_end_mmdd[0]  , melt_end_mmdd[1])

    # Pull out all the datetimes and melt_data from within the baseline melt seasons.
    dt_list_melt_season = []
    for i,year in enumerate(range(baseline_start_year, baseline_end_year+(0 if (melt_start_mmdd > melt_end_mmdd) else 1))):
        year_start_dt = datetime.datetime(year=year, month=melt_start_mmdd[0], day=melt_start_mmdd[1])
        year_end_dt = datetime.datetime(year=(year+1 if (melt_start_mmdd > melt_end_mmdd) else year), month=melt_end_mmdd[0], day=melt_end_mmdd[1])

        dt_mask = [((dt >= year_start_dt) and (dt <= year_end_dt)) for dt in datetimes_list]
        dt_list_melt_season.extend([dt for i,dt in enumerate(datetimes_list) if dt_mask[i]])

        year_data = melt_array_nan_filled[:,:,numpy.array(dt_mask)]

        if i==0:
            melt_season_array_nan_filled = year_data
        else:
            melt_season_array_nan_filled = numpy.concatenate( (melt_season_array_nan_filled, year_data), axis=2)

    # Get a list of every (month,day) tuple from the melt_start_mmdd to the melt_end_mmdd.
    # Use the year 1999-2000 in order to include an entry for February 29th.
    filler_start_date = datetime.datetime(1999 if melt_start_mmdd > melt_end_mmdd else 2000, melt_start_mmdd[0], melt_start_mmdd[1])
    filler_end_date   = datetime.datetime(2000, melt_end_mmdd[0], melt_end_mmdd[1])

    # Get the total number of days we want to get data for
    num_days = (filler_end_date - filler_start_date).days

    # Get a list of datetime objects for each of those days
    baseline_filler_dt_list = [filler_start_date + datetime.timedelta(days=d) for d in range(num_days+1)]

    # # A vector of the months, and days of each datetime, for easy searching.
    # baseline_dt_list_months = numpy.array([dt.month for dt in dt_list_melt_season], dtype=numpy.uint8)
    # baseline_dt_list_day_of_months = numpy.array([dt.day for dt in dt_list_melt_season], dtype=numpy.uint8)

    # Generate an empty MxNxT array with
    average_melt_array = numpy.zeros((melt_array.shape[0], melt_array.shape[1], len(baseline_filler_dt_list)), dtype=numpy.float)

    # Now, compute the average odds (0-1) of melt on any given day for any given pixel, over the baseline period.
    for i,bdt in enumerate(baseline_filler_dt_list):
        bdt_day_mask = numpy.array( [((dt.month == bdt.month) and (dt.day == bdt.day)) for dt in dt_list_melt_season] , dtype=numpy.bool)
        # # print ("\t", [dt for i,dt in enumerate(dt_list_melt_season) if bdt_day_mask[i]])
        # if numpy.count_nonzero(bdt_day_mask) == 0:
        #     print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        melt_array_day_slice = melt_season_array_nan_filled[:,:,bdt_day_mask]

        # print (i, bdt, numpy.count_nonzero(bdt_day_mask), melt_array_day_slice.shape)

        # if melt_array_day_slice.shape[2] == 0:
        #     print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        average_melt_array[:,:,i] = numpy.nanmean(melt_array_day_slice, axis=2, dtype=numpy.float32)

        # print(average_melt_array[:,:,i].size, "total,",
        #       numpy.count_nonzero(numpy.isnan(average_melt_array[:,:,i])), "nans,",
        #       numpy.count_nonzero(average_melt_array[:,:,i] == 0), "zeros,",
        #       numpy.count_nonzero(average_melt_array[:,:,i]) - numpy.count_nonzero(numpy.isnan(average_melt_array[:,:,i])), "non-zero.")

    # Fill in "nan" values with zero (if they're in the mask)
    # Fill in "nan" values outside the mask with -1.0 (just look at the original array for these)
    # Don't leave any nans in there.
    average_melt_array[melt_array[:,:,0]==-1,:] = -1
    # print(numpy.count_nonzero(numpy.isnan(average_melt_array)), "of", average_melt_array.size, "are nans.", numpy.count_nonzero(numpy.isnan(average_melt_array))/average_melt_array.shape[2], "average.")
    average_melt_array[numpy.isnan(average_melt_array)] = 0

    # Quick check of the histogram (comment-out once we've done it)
    # import matplotlib
    # matplotlib.pyplot.hist(average_melt_array.flatten(), bins=30)
    # matplotlib.pyplot.ylim([0,1e4])
    # matplotlib.pyplot.show()

    # Create a dictionary with (mm,dd) keys and i-values to indext into the average_melt_array
    baseline_dates_mmdd_dict = dict([((dt.month, dt.day),i) for i,dt in enumerate(baseline_filler_dt_list)])

    # Save out to a picklefile array.
    if output_picklefile != None:
        f = open(output_picklefile, 'wb')
        pickle.dump((average_melt_array, baseline_dates_mmdd_dict), f)
        f.close()
        if verbose:
            print(output_picklefile, "written.")

    return average_melt_array, baseline_dates_mmdd_dict

def read_daily_melt_averages_picklefile(build_picklefile_if_not_present=True,
                                        daily_climatology_picklefile=daily_melt_averages_picklefile,
                                        verbose=True):
    """Read the daily climatology averages picklefile."""
    if not os.path.exists(daily_climatology_picklefile):
        if build_picklefile_if_not_present:
            return compute_daily_climatology_pixel_averages(output_picklefile=daily_climatology_picklefile,
                                                            verbose=verbose)
        else:
            raise FileNotFoundError("Picklefile '{0}' not found.".format(daily_climatology_picklefile))

    if verbose:
        print("Reading", daily_climatology_picklefile)
    f = open(daily_climatology_picklefile, 'rb')
    array, dt_dict = pickle.load(f)
    f.close()

    return array, dt_dict

def compute_daily_sum_pixel_averages(daily_picklefile=daily_melt_averages_picklefile,
                                     sum_picklefile=daily_cumulative_melt_averages_picklefile,
                                     verbose=True):
    """Compute a mean daily cumulative melt-day value for each pixel throughout the melt season.

    {(mm,dd):(MxN array of integer melt days)}

    Just like the (mm,dd):(MxN array) daily melt averages picklefile, this sums up the total for each days,
    giving a cumulative melt picklefile for the same data. This tells, for each pixel in a "mean" year,
    how much melt there exists for each day of the melt season, in cumulative days.

    The daily melt total are in floating point, but the sum is given in integers, so as to be
    directly comparable to a given daily-sum value during the melt season.
    """
    # First, read the daily melt value picklefile.
    daily_array, dt_dict = read_daily_melt_averages_picklefile(verbose=verbose)
    daily_sum_array = numpy.zeros(daily_array.shape, dtype=numpy.int32)
    for dt in dt_dict:
        daily_sum_array[:,:,dt_dict[dt]] = numpy.array(numpy.round(numpy.sum(daily_array[:,:,0:dt_dict[dt]], axis=2), decimals=0), dtype=numpy.int32)

    if verbose:
        print("Writing", sum_picklefile, end="...")
    f = open(sum_picklefile, 'wb')
    pickle.dump((daily_sum_array, dt_dict), f)
    f.close()
    if verbose:
        print("Done.")

def read_daily_sum_melt_averages_picklefile(build_picklefile_if_not_present=True,
                                            daily_sum_picklefile=daily_cumulative_melt_averages_picklefile,
                                            verbose=True):
    """Read the daily climatology averages picklefile."""
    if not os.path.exists(daily_sum_picklefile):
        if build_picklefile_if_not_present:
            return compute_daily_climatology_pixel_averages(output_picklefile=daily_sum_picklefile,
                                                            verbose=verbose)
        else:
            raise FileNotFoundError("Picklefile '{0}' not found.".format(daily_sum_picklefile))

    if verbose:
        print("Reading", daily_sum_picklefile)
    f = open(daily_sum_picklefile, 'rb')
    array, dt_dict = pickle.load(f)
    f.close()

    return array, dt_dict


def create_baseline_climatology_tif(start_date = datetime.datetime(1990,10,1),
                                end_date = datetime.datetime(2020,4,30),
                                f_out_mean = mean_climatology_geotiff,
                                f_out_std = std_climatology_geotiff,
                                round_to_integers = True,
                                gap_filled = True,
                                verbose=True):
    """Generate a "mean annual melt" map over the baseline period.

    The melt year for each season is defined from the (mm,dd) from the "start_date"
    to the (mm,dd) from the "end_date" of the following year (assuming it wraps
    around the year).
    """
    # Read the gridded satellite data
    if gap_filled:
        model_array, datetimes_dict = read_gap_filled_melt_picklefile(verbose=verbose)
    else:
        model_array, datetimes_dict = read_model_array_picklefile(resample_melt_codes = True, verbose=verbose)
    datetimes = list(datetimes_dict.keys())

    num_years = int((end_date - start_date).days / 365.25)
    # print(num_years)

    annual_sum_grids = numpy.empty(model_array.shape[0:2]+(num_years,), dtype=numpy.int)

    if gap_filled:
        model_melt_days = model_array
    else:
        model_melt_days = (model_array == 2)

    ice_mask = get_ice_mask_array()

    wrap_year = (start_date.month, start_date.day) > (end_date.month, end_date.day)

    for i in range(num_years):
        dt1 = datetime.datetime(start_date.year + i, start_date.month, start_date.day)
        dt2 = datetime.datetime(start_date.year + i + (1 if wrap_year else 0), end_date.month  , end_date.day)

        # print(i, dt1, dt2)

        dates_mask = numpy.array([(dt >= dt1) & (dt <= dt2) for dt in datetimes], dtype=numpy.bool)

        # dt1_i = datetimes.index(dt1)
        # dt2_i = datetimes.index(dt2)

        annual_sum_grids[:,:,i] = numpy.sum(model_melt_days[:,:,dates_mask], axis=2)

    annual_mean_array = numpy.mean(annual_sum_grids, axis=2, dtype=numpy.float32)
    annual_std_array = numpy.std(annual_sum_grids, axis=2, dtype=numpy.float32)

    if round_to_integers:
        annual_mean_array = numpy.array(numpy.round(annual_mean_array), dtype=numpy.int)
        annual_std_array = numpy.array(numpy.round(annual_std_array), dtype=numpy.int)

    annual_mean_array[(ice_mask==0)] = -1
    annual_std_array [(ice_mask==0)] = -1

    # If we're using the gap-filled data, add "gap_filled" to the name.
    if gap_filled:
        f_mean_base, f_mean_ext = os.path.splitext(f_out_mean)
        f_out_mean = f_mean_base + "_gap_filled" + f_mean_ext

        f_std_base, f_std_ext = os.path.splitext(f_out_std)
        f_out_std = f_std_base + "_gap_filled" + f_std_ext

    output_gtif(annual_mean_array, f_out_mean, nodata=-1, verbose=verbose)
    output_gtif(annual_std_array, f_out_std, nodata=-1, verbose=verbose)

    return annual_mean_array

def create_partial_year_melt_anomaly_tif(current_datetime=None,
                                         dest_fname=None,
                                         gap_filled=True,
                                         verbose=True):
    """Create a tif of melt anomlay compared to baseline climatology for that day of the melt season."""
    # If no datetime is given, use "today"
    if current_datetime is None:
        now = datetime.datetime.today()
        # Strip of the hour,min,second
        current_datetime = datetime.datetime(year=now.year, month=now.month, day=now.day)

    daily_melt_sums, daily_sums_dt_dict = read_daily_sum_melt_averages_picklefile()

    first_mmdd_of_melt_season = list(daily_sums_dt_dict.keys())[0]
    start_year_of_current_melt_season = current_datetime.year + (0 if first_mmdd_of_melt_season < (current_datetime.month, current_datetime.year) else -1)

    first_dt_of_present_melt_season = datetime.datetime(year=start_year_of_current_melt_season,
                                                        month=first_mmdd_of_melt_season[0],
                                                        day=first_mmdd_of_melt_season[1])

    # print(current_datetime)
    # print(first_dt_of_present_melt_season)
    if gap_filled:
        melt_array, dt_dict = read_gap_filled_melt_picklefile(verbose=verbose)
    else:
        melt_array, dt_dict = read_model_array_picklefile(resample_melt_codes=True, verbose=verbose)

    dt_list = sorted(list(dt_dict.keys()))
    dt_mask = numpy.array([((dt >= first_dt_of_present_melt_season) and (dt <= current_datetime)) for dt in dt_list], dtype=numpy.bool)
    dts_masked = [dt for dt,mask_val in zip(dt_list, dt_mask) if mask_val]

    # If we don't have days in the picklefile up to the current date, readjust the date and inform the user.
    if dts_masked[-1] < current_datetime:
        print("{0} not in the melt files. Adjusting to last known date: {1}".format(current_datetime.strftime("%Y-%m-%d"),
                                                                                    dts_masked[-1].strftime("%Y-%m-%d")))
        current_datetime = dts_masked[-1]

    current_season_melt_slice = melt_array[:,:,dt_mask]
    if gap_filled:
        sum_melt_days_current_season = numpy.sum(current_season_melt_slice, axis=2)
    else:
        sum_melt_days_current_season = numpy.sum(current_season_melt_slice == 2, axis=2)

    avg_melt_days_for_this_mmdd = daily_melt_sums[:,:,daily_sums_dt_dict[(current_datetime.month, current_datetime.day)]]

    anomaly_this_season_so_far = sum_melt_days_current_season - avg_melt_days_for_this_mmdd

    ice_mask = get_ice_mask_array()
    anomaly_this_season_so_far[ice_mask==0] = -999

    # Round to integers, if it isn't already.
    anomalies_int = numpy.array(numpy.round(anomaly_this_season_so_far), dtype=numpy.int32)

    # If dest_fname is None, create it.
    if dest_fname is None:
        dest_fname = os.path.join(os.path.split(outputs_annual_tifs_directory)[0], "annual_anomalies",
                        "{0}_anomaly{1}.tif".format(current_datetime.strftime("%Y.%m.%d"), "_gap_filled" if gap_filled else ""))

    output_gtif(anomalies_int, dest_fname, nodata=-999, verbose=verbose)

    return anomalies_int

def create_annual_melt_anomaly_tif(year,
                                   year_melt_tif = None,
                                   baseline_melt_tif = None,
                                   gap_filled = True,
                                   verbose=True):
    """Create a tif of annual melt anomaly compared to baseline climatology.

    If None are selected for year_melt_tif or baseline_melt_tif, generate those files.
    """
    dest_fname = os.path.join(os.path.split(outputs_annual_tifs_directory)[0], "annual_anomaly_geotifs",
                            "{0}-{1}_anomaly{2}.tif".format(year, year+1, "_gap_filled" if gap_filled else ""))

    year_array = get_annual_melt_sum_array(year, fname=year_melt_tif, gap_filled=gap_filled)
    if year_array is None:
        return None

    mean_array = get_baseline_climatology_array(fname = baseline_melt_tif, gap_filled=gap_filled)

    anomaly_array = year_array - mean_array

    ice_mask = get_ice_mask_array()

    anomaly_array[ice_mask==0] = -999

    output_gtif(anomaly_array, dest_fname, verbose=verbose, nodata=-999)

    return anomaly_array


def read_annual_melt_anomaly_tif(year,
                                 anomaly_tif=None,
                                 gap_filled=True,
                                 generate_if_nonexistent=True,
                                 verbose=True):
    """Read the annual anomaly tif."""
    if anomaly_tif is None:
        anomaly_tif = os.path.join(os.path.split(outputs_annual_tifs_directory)[0], "annual_anomalies",
                            "{0}-{1}_anomaly{2}.tif".format(year, year+1, "_gap_filled" if gap_filled else ""))

    if not os.path.exists(anomaly_tif):
        array = create_annual_melt_anomaly_tif(year=year,
                                               gap_filled=gap_filled,
                                               verbose=verbose)
        return array

    if verbose:
        print("Reading", anomaly_tif)

    ds = gdal.Open(anomaly_tif,gdal.GA_ReadOnly)
    if ds is None:
        return None

    array = ds.GetRasterBand(1).ReadAsArray()

    return array


def get_baseline_climatology_array(fname=None,gap_filled=True):
    """Retreive the melt year array from the tif.

    If it's not available, create it and write the file, then return it.
    """
    if fname == None:
        fname = mean_climatology_geotiff

    if gap_filled and os.path.split(fname)[1].find("gap_filled") == -1:
        base, ext = os.path.splitext(fname)
        fname = base + "_gap_filled" + ext

    if os.path.exists(fname):
        ds = gdal.Open(fname,gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError("Could not open {0}".format(fname))
        array = ds.GetRasterBand(1).ReadAsArray()
        return array

    return create_baseline_climatology_tif(gap_filled=gap_filled)

def get_annual_melt_sum_array(year, fname=None, gap_filled=True):
    """Retreive the melt year array from the tif.

    If it's not available, create it and write the file, then return it.
    """
    if fname is None:
        fname = os.path.join(outputs_annual_tifs_directory, "{0}-{1}.tif".format(year, year+1))

    if gap_filled:
        base, ext = os.path.splitext(fname)
        fname = base + "_gap_filled" + ext

    if os.path.exists(fname) and os.path.split(fname)[1].find("gap_filled") == -1:
        ds = gdal.Open(fname,gdal.GA_ReadOnly)
        array = ds.GetRasterBand(1).ReadAsArray()
        return array

    return create_annual_melt_sum_tif(year=year, gap_filled=gap_filled)

def create_annual_melt_sum_tif(year = "all",
                               output_tif = None,
                               melt_start_mmdd = (10,1),
                               melt_end_mmdd = (4,30),
                               gap_filled = True,
                               verbose=True):
    """Create an integer tif file of that year's annual sum of melt-days, per pixel.

    If gap_filled, create a floating-point tif file of the same.

    This is just from the raw data, no corrections (yet) are performed on it. See about
    changing that.
    """
    if gap_filled:
        melt_array, datetimes_dict = read_gap_filled_melt_picklefile(verbose=verbose)
    else:
        melt_array, datetimes_dict = read_model_array_picklefile(resample_melt_codes = True,
                                                                 verbose=verbose)
    dt_list = list(datetimes_dict.keys())

    if year == "all":
        years = numpy.unique([dt.year for dt in dt_list])
        years.sort()

    else:
        assert year == int(year)
        years = [year]

    ice_mask = get_ice_mask_array()

    melt_array_year = None

    for y in years:
        start_date = datetime.datetime(year=y, month=melt_start_mmdd[0], day=melt_start_mmdd[1])
        end_date = datetime.datetime(year=(y if (melt_start_mmdd < melt_end_mmdd) else y+1),
                                     month=melt_end_mmdd[0], day=melt_end_mmdd[1])

        dates_mask = numpy.array([((dt >= start_date) and (dt <= end_date)) for dt in dt_list], dtype=numpy.bool)

        # Skip any years for which there is no data.
        if numpy.count_nonzero(dates_mask) == 0:
            continue

        if gap_filled:
            # First add the floating-point values
            melt_array_year = numpy.sum(melt_array[:,:,dates_mask], axis=2, dtype=numpy.float)
            # Round to integers
            melt_array_year = numpy.array(melt_array_year.round(), dtype=numpy.int)
        else:
            melt_array_year = numpy.sum((melt_array[:,:,dates_mask] == 2), axis=2, dtype=numpy.int)

        melt_array_year[ice_mask == 0] = -1

        if output_tif is None:
            fname = "{0}-{1}.tif".format(y, y + 1) if (melt_start_mmdd > melt_end_mmdd) else "{0}.tif".format(y)
            output_fname = os.path.join(outputs_annual_tifs_directory, fname)
        else:
            output_fname = output_tif

        if gap_filled:
            base, ext = os.path.splitext(output_fname)
            output_fname = base + "_gap_filled" + ext

        output_gtif(melt_array_year, output_fname, nodata=-1, verbose=verbose)

    return melt_array_year

def save_climatologies_as_CSV(csv_file = baseline_percentiles_csv,
                              baseline_start_year = 1990,
                              baseline_end_year = 2020,
                              doy_start = (10,1),
                              doy_end = (4,30),
                              gap_filled = True,
                              verbose=True):
    """Compute the percentiles of climatologies and save them as a pandas dataframe for later use."""
    baseline_melt_percentiles_for_each_basin = \
        _generate_baseline_melt_climatology(baseline_start_year = baseline_start_year,
                                            baseline_end_year = baseline_end_year,
                                            doy_start = doy_start,
                                            doy_end = doy_end,
                                            include_regional_totals= True,
                                            gap_filled = gap_filled,
                                            verbose=verbose)

    assert len(baseline_melt_percentiles_for_each_basin) == len(antarctic_regions_dict)

    text_lines = ["##########################################################",
                  "# Antarctica Today Baseline Melt Climatology",
                  "# Baseline years: October 1, 1980 thru April 30, 2011"
                  "# Climatologies computed for each day of the 'melt season', October 1 thru April 30 for that 30 year baseline record.",
                  "# Areas are in km2, fractions are fractions of a whole (0 to 1), based upon available data for each measured day in the dataset."
                  "##########################################################",
                  "# Author: Dr. Michael MacFerrin, University of Colorado",
                  "# Date generated: {0}".format(datetime.date.today().strftime("%Y-%m-%d")),
                  "##########################################################",
                  "# Region Numbers:"]
    for region_num in range(0,len(baseline_melt_percentiles_for_each_basin)):
        # Generate the CSV header listing the region numbers.
        text_lines.append("#     R{0}: {1}".format(region_num, antarctic_regions_dict[region_num]))

    text_lines.append("##########################################################")
    text_lines.append("# See README and {0} for details on region areas and outlines.".format(os.path.split(antarctic_regions_tif)[1]))
    text_lines.append("##########################################################")

    csv_fields_line = "month,day,"
    for region_num in range(len(baseline_melt_percentiles_for_each_basin)):
        # Generate the csv header fields for that region (R0 thru R7)
        # R0_area_min,R0_area_10,R0_area_25,R0_area_50,R0_area_75,R0_area_90,R0_area_max,...
        # R0_fraction_min,R0_fraction_10,R0_fraction_25,R0_fraction_50,R0_fraction_75,R0_fraction_90,R0_fraction_max,...
        # R1_area_min, ... (etc)

        data_segment_line = \
            "R{0}_area_min,R{0}_area_10,R{0}_area_25,R{0}_area_50,R{0}_area_75,R{0}_area_90,R{0}_area_max," + \
            "R{0}_fraction_min,R{0}_fraction_10,R{0}_fraction_25,R{0}_fraction_50,R{0}_fraction_75,R{0}_fraction_90,R{0}_fraction_max" + \
            ("," if (region_num < (len(baseline_melt_percentiles_for_each_basin)-1)) else "")

        data_segment_line = data_segment_line.format(region_num)

        csv_fields_line = csv_fields_line + data_segment_line


    text_lines.append(csv_fields_line)

    # Sort from doy_start to doy_end, wrap around the new year
    month_day_tuples = baseline_melt_percentiles_for_each_basin[0].keys()
    if doy_start > doy_end:
        md_tuples_start = [md for md in month_day_tuples if (md >= doy_start)]
        md_tuples_end   = [md for md in month_day_tuples if (md < doy_start)]
        month_day_tuples = md_tuples_start + md_tuples_end


    for md_tuple in month_day_tuples:
        csv_line = "{0},{1}".format(md_tuple[0], md_tuple[1])

        # For each region, Get all the pixelcounts (convert to areas) and fractions, and load them in the CSV.

        for region_num in range(len(baseline_melt_percentiles_for_each_basin)):
            melt_pixels, melt_fractions = baseline_melt_percentiles_for_each_basin[region_num][md_tuple]
            melt_stats = list(melt_pixels*(25**2)) + list(melt_fractions)
            stats_str = "," + ",".join([str(n) for n in melt_stats])

            csv_line = csv_line + stats_str

        text_lines.append(csv_line)

    text_all = "\n".join(text_lines)

    if gap_filled:
        base, ext = os.path.splitext(csv_file)
        csv_file = base + "_gap_filled" + ext

    # Save the CSV.
    with open(csv_file, 'w') as f:
        f.write(text_all)

    if verbose:
        print(csv_file, "written.")

    return

def _generate_baseline_melt_climatology(baseline_start_year = 1990,
                                        baseline_end_year   = 2020,
                                        doy_start     = (10,1), # (MM,DD)
                                        doy_end       = (4,30), # (MM,DD)
                                        include_regional_totals = True,
                                        gap_filled = True,
                                        verbose=True):
    """Generate the data for a climatology plot.

    baseline_start_year and baseline_end_year:
        Years to begin and end the baseline climate period. Defaults to October 1990 through April 2020.

    doy_start, doy_end:
        (month,day) tuples to delineate the start and end of the "melt year".
        Defaults to (10,1) and (4,30), respectively (Oct 1 thru April 30).

    percent_or_area:
        plot the y-axis by percentage, or total area (km2).
        Possible values: "percent" or "area". Any other values will throw an error.

    include_regional_plots:
        If False, just plot the entire continent climatology.
        If True, plot each of the sub-regions as well.
    """
    if gap_filled:
        melt_array, datetime_dict = read_gap_filled_melt_picklefile(verbose=verbose)
    else:
        melt_array, datetime_dict = read_model_array_picklefile(fill_pole_hole=True,
                                                                resample_melt_codes=True,
                                                                resample_melt_code_threshold=4,
                                                                verbose=verbose)

    ice_mask = get_ice_mask_array()

    # Build a list of datetimes in the year 2000, which includes Feb 29th, to get a list of all the (month,day) pairs we need.
    # datetimes_2000 = [datetime.datetime.strptime(dstr, "%Y%j") for dstr in ["2000{0:03d}".format(d) for d in range(1,367)]]
    # # Generate a list of all (month,day) tuples in the year
    # if doy_start > doy_end:
    #     month_day_tuples = [(dt.month, dt.day) for dt in datetimes_2000 if ((dt.month,dt.day) >= doy_start) or ((dt.month,dt.day) <= doy_end)]
    # else:
    #     month_day_tuples = [(dt.month, dt.day) for dt in datetimes_2000 if (doy_start <= (dt.month,dt.day) <= doy_end)]
    # # Build a list of datetimes in the year 1999-2000, which includes Feb 29th. Then put all our datetimes into it.
    # month_day_tuples.sort()

    # Gather all the masks, regional included (if requested)
    ice_masks = [ice_mask]
    if include_regional_totals:
        regional_masks_dict = _get_regional_tif_masks()
        for i in range(1,8):
            ice_masks.append(regional_masks_dict[i])
            # print(i,antarctic_regions_dict[i], numpy.sum(ice_masks[i]))

    percentiles_dict_list = [None] * len(ice_masks)
    for i, mask_array in enumerate(ice_masks):
        melt_day_percentiles_dict = _compute_baseline_melt_percentiles(melt_array,
                                                                       datetime_dict,
                                                                       mask_array,
                                                                       baseline_start_year,
                                                                       baseline_end_year,
                                                                       doy_start,
                                                                       doy_end,
                                                                       gap_filled=gap_filled)

        percentiles_dict_list[i] = melt_day_percentiles_dict

    return percentiles_dict_list

def open_baseline_climatology_csv_as_dataframe(csv_file = baseline_percentiles_csv,
                                               gap_filled=True,
                                               verbose = True):
    """Open the dataframe for the baseline period climatology percentiles, and return a pandas dataframe."""
    if gap_filled and os.path.split(csv_file)[1].find("gap_filled") == -1:
        base, ext = os.path.splitext(csv_file)
        csv_file = base + '_gap_filled' + ext

    if verbose:
        print("Reading", csv_file)
    return pandas.read_csv(csv_file, header=19)


def _get_regional_tif_masks(tifname = antarctic_regions_tif):
    """Read the tif listing all the region identifiers in Antarctica.

    Return:
    ------
    - A (region:mask) dictionary for each region in the tif file.
    Regions will be 1..7. masks will each be an MxN array of (0,1) values.
    """
    # Read the ice mask tif, return the array.
    regions_ds = gdal.Open(tifname, gdal.GA_ReadOnly)
    region_array = regions_ds.GetRasterBand(1).ReadAsArray()
    ndv = regions_ds.GetRasterBand(1).GetNoDataValue()
    unique_vals = [val for val in numpy.unique(region_array) if val != ndv]

    # Sanity check:
    # Make sure the total of the masks is the same area covered by the ice mask.
    ice_mask = get_ice_mask_array()
    try:
        assert numpy.all((ice_mask > 0) == (region_array > 0))
    except AssertionError as e:
        print("ice_mask:", numpy.count_nonzero(ice_mask>0))
        print("regions: ", numpy.count_nonzero(region_array>0))
        raise e

    output_mask_dict = {}
    for i in unique_vals:
        output_mask_dict[i] = (region_array == i)

    return output_mask_dict

def _get_region_area_km2(region_number,
                         resolution_km = 25):
    """Given a particular region number, compute the total area in km2 of the region.

    If you want to get the pixel count, simply set "resolution_km" to 1.
    """
    region_mask_dict = _get_regional_tif_masks()
    return numpy.count_nonzero(region_mask_dict[region_number]) * (resolution_km**2)

def _compute_baseline_melt_percentiles(melt_array,
                                       datetimes_dict,
                                       mask_array,
                                       baseline_start_year,
                                       baseline_end_year,
                                       doy_start,
                                       doy_end,
                                       gap_filled=True):
    """Compute the (min,10,25,50,75,90th,max) percentile value in each of the days from doy_start to doy_end (MM,DD), during the baseline years.

    Return:
    ------
    A dictionary with (month,day) tuple keys, and 5-element vector values listing the 10,25,50,75,90th percentiles of each melt day.
    """
    melt_day_dict = _compute_baseline_climatology_lists(melt_array,
                                                        datetimes_dict,
                                                        mask_array,
                                                        baseline_start_year,
                                                        baseline_end_year,
                                                        doy_start,
                                                        doy_end,
                                                        gap_filled=gap_filled)

    output_dict = {}

    for dm_tuple in melt_day_dict.keys():
        melt_pixel_list = melt_day_dict[dm_tuple][0]
        melt_fraction_list = melt_day_dict[dm_tuple][1]
        # For each (month,day) tuple, compute a 2-tuples of lists (pixels, fraction).
        # Each list contains (min, 10%, 25%, 50%, 75%, 90%, max)
        output_dict[dm_tuple] = (numpy.array([numpy.min(melt_pixel_list)] + list(numpy.percentile(melt_pixel_list, (10,25,50,75,90))) + [numpy.max(melt_pixel_list)]),
                                 numpy.array([numpy.min(melt_fraction_list)] + list(numpy.percentile(melt_fraction_list, (10,25,50,75,90))) + [numpy.max(melt_fraction_list)]))

    return output_dict

def _compute_baseline_climatology_lists(melt_array,
                                        datetimes_dict,
                                        mask_array,
                                        baseline_start_year,
                                        baseline_end_year,
                                        doy_start,
                                        doy_end,
                                        gap_filled=True):
    """Compute the annual list of melt pixels and percentages on a day-of-year basis.

    Parameters
    ----------
        melt_array: MxNxT daily gridded melt values: 0: no data, 1: no melt, 2: melt

        datetimes_dict: dictionary of (datetime:index) pairs along the T-axis

        mask_array: the mask to use for the selected region of the continent.
                    Usually either the whole area mask, or a regional mask.

        baseline_start_year: The year to start the baseline record

        baseline_end_year: The year to end the baseline record

        doy_start: The (month,day) tuple to start the melt year

        doy_end:   The (month,day) tuple to end the melt year.

    Return:
    ------
        - A dictionary with (month,day) keys and (a 2-length tuples of lists of cumaltive melt pixels over all the years) values.
        It the tuple, the first list in the melt pixels, the second list is the percentages.
        For instance, if April 30 had 30 years of melt in the dataset, it would be
        a (4,30):([30 individual pixel counts],[30 individual fractions]) pair in the dictionary. Only days from
        doy_start to doy_end will be included.  If doy_start > doy_end (as it will be in Antarctica),
        then it will wrap around December but not include mid-year (austral winter) months in between.

    NOTE: Not all days-of-year will have the same length list. For instance,
    Feb 29 will only happen 1-in-4 yeras, and some years are missing occasional days
    in the climatology (especially in the 1980s)

    """
    # Build a list of datetimes in the year 2000, which includes Feb 29th, to build our dictionary.
    datetimes_2000 = [datetime.datetime.strptime(dstr, "%Y%j") for dstr in ["2000{0:03d}".format(d) for d in range(1,367)]]

    mask_array_broadcast = numpy.copy(mask_array)
    mask_array_broadcast.shape = tuple(list(mask_array.shape) + [1])
    # Add up the melt pixels for each day in the array, over the mask:

    if gap_filled:
        melt_vector = numpy.sum(melt_array * mask_array_broadcast, axis=(0,1))
    else:
        melt_vector = numpy.sum(((melt_array == 2) * mask_array_broadcast), axis=(0,1))

    total_pixel_vector = numpy.sum(((melt_array > -1) * mask_array_broadcast), axis=(0,1))

    # Build the output dictionary with empty lists to start with:
    if doy_start > doy_end:
        output_dict = dict([((dt.month, dt.day), [[],[]]) for dt in datetimes_2000 if ((dt.month,dt.day) >= doy_start) or ((dt.month,dt.day) <= doy_end)])
    else:
        output_dict = dict([((dt.month, dt.day), [[],[]]) for dt in datetimes_2000 if (doy_start <= (dt.month,dt.day) <= doy_end)])

    for dt in datetimes_dict.keys():
        # If it's outside our baseline span, just omit and move along.
        if (dt < datetime.datetime(baseline_start_year, *doy_start)) \
            or (dt > datetime.datetime(baseline_end_year, *doy_end)):
            continue

        # If it's not within our melt year span (all the days of which are
        # included in the dictionary already), then move along to the next day.
        if (dt.month, dt.day) not in output_dict:
            continue

        # Add the total melt pixels for that day-of-year to the list already in the dictionary.
        melt_list_for_that_day, fraction_list_for_that_day = output_dict[(dt.month, dt.day)]

        if total_pixel_vector[datetimes_dict[dt]] > 0:
            melt_list_for_that_day.append(melt_vector[datetimes_dict[dt]])
            fraction_list_for_that_day.append(float(melt_vector[datetimes_dict[dt]]) / float(total_pixel_vector[datetimes_dict[dt]]))

        output_dict[(dt.month, dt.day)] = (melt_list_for_that_day, fraction_list_for_that_day)

    return output_dict


def read_daily_melt_numbers_as_dataframe(csv_file=daily_melt_csv,
                                         gap_filled=True,
                                         verbose=True):
    """Read the daily melt files, return a Pandas dataframe."""
    if gap_filled and os.path.split(csv_file)[1].find("gap_filled") == -1:
        base, ext = os.path.splitext(csv_file)
        csv_file = base + "_gap_filled" + ext

    if verbose:
        print("Reading", csv_file)

    # Read the dataframe. Conver the "date" field to a date object.
    df = pandas.read_csv(csv_file, header=18, converters={"date":pandas.to_datetime})

    return df

def save_daily_melt_numbers_to_csv(csv_file=daily_melt_csv,
                                   gap_filled=True,
                                   verbose=True):
    """Compute climatologies for all regions on every day of the dataset, save to a .csv file."""
    text_lines = ["##########################################################",
                  "# Antarctica Today Daily Melt Calculations",
                  "# Areas are in km2, fractions are fractions of a whole (0 to 1), based upon available data for each measured day in the dataset."
                  "##########################################################",
                  "# Author: Dr. Michael MacFerrin, University of Colorado",
                  "# Date generated: {0}".format(datetime.date.today().strftime("%Y-%m-%d")),
                  "##########################################################",
                  "# Region Numbers:"]
    for region_num in range(0,len(antarctic_regions_dict)):
        # Generate the CSV header listing the region numbers.
        text_lines.append("#     R{0}: {1}".format(region_num, antarctic_regions_dict[region_num]))

    text_lines.append("##########################################################")
    text_lines.append("# See README and {0} for details on region areas and outlines.".format(os.path.split(antarctic_regions_tif)[1]))
    text_lines.append("##########################################################")

    csv_fields_line = "date,"
    for region_num in range(len(antarctic_regions_dict)):
        # Generate the csv header fields for that region (R0 thru R7)
        # R0_good_data_area, R0_melt_area, R0_melt_fraction, R0_fraction_complete,
        # R1_good_data_area....

        data_segment_line = \
            "R{0}_good_data_area,R{0}_melt_area,R{0}_melt_fraction,R{0}_fraction_complete" + \
            ("," if (region_num < (len(antarctic_regions_dict)-1)) else "")

        data_segment_line = data_segment_line.format(region_num)

        csv_fields_line = csv_fields_line + data_segment_line

    text_lines.append(csv_fields_line)

    if gap_filled:
        melt_array, datetime_dict = read_gap_filled_melt_picklefile(verbose=False)
    else:
        melt_array, datetime_dict = read_model_array_picklefile(fill_pole_hole=True,
                                                            filter_out_error_swaths=True,
                                                            resample_melt_codes=True,
                                                            resample_melt_code_threshold=4,
                                                            verbose=False
                                                            )

    ice_mask = get_ice_mask_array()

    # Gather all the masks, regional included
    ice_masks = [ice_mask]
    regional_masks_dict = _get_regional_tif_masks()
    for i in range(1,len(antarctic_regions_dict)):
        ice_masks.append(regional_masks_dict[i])

    # Compute the melt for all the days.
    good_data_area_vectors = {}
    melt_area_vectors = {}
    melt_fraction_vectors = {}
    fraction_complete_vectors = {}
    for i, mask_array in enumerate(ice_masks):
        mask_area_total = numpy.count_nonzero(mask_array) * (25**2)
        # Add a dimension to the mask shape to broadcast to 3D
        mask_array.shape = tuple(list(mask_array.shape) + [1])

        # Add up the melt pixels for each day in the array, over the mask:
        if gap_filled:
            melt_pixel_vector = numpy.sum((melt_array * mask_array), axis=(0,1))
            total_good_pixel_vector = numpy.sum(((melt_array > -1) * mask_array), axis=(0,1))
        else:
            melt_pixel_vector = numpy.sum(((melt_array == 2) * mask_array), axis=(0,1))
            total_good_pixel_vector = numpy.sum((((melt_array == 2) | (melt_array == 1)) * mask_array), axis=(0,1))

        good_data_area_vectors[i] = total_good_pixel_vector * (25**2)
        fraction_complete_vectors[i] = good_data_area_vectors[i] / mask_area_total
        melt_area_vectors[i] = melt_pixel_vector * (25**2)
        # TODO: Test whether this works.
        # Fix warning here: "RuntimeWarning: invalid value encountered in true_divide.
        # Obviously we have zome zero good_data_area_vector values here, probably early in the dataset.
        # Set the denominator to one.
        good_data_area_vectors[i][total_good_pixel_vector == 0] = 1.0

        assert numpy.all(good_data_area_vectors[i] >= melt_area_vectors[i])

        melt_fraction_vectors[i] = melt_area_vectors[i] / good_data_area_vectors[i]

    for dt in datetime_dict.keys():
        # Get each date.
        data_line = "{0},".format(dt.strftime("%Y-%m-%d"))
        dt_i = datetime_dict[dt]
        # Get the stats on that date for each region.
        for region_num in antarctic_regions_dict.keys():
            data_items_template = "{0},{1},{2},{3}" + ("," if (region_num < (len(antarctic_regions_dict)-1)) else "")

            data_items_segment = data_items_template.format(good_data_area_vectors[region_num][dt_i],
                                                            melt_area_vectors[region_num][dt_i],
                                                            melt_fraction_vectors[region_num][dt_i],
                                                            fraction_complete_vectors[region_num][dt_i])

            data_line = data_line + data_items_segment

        text_lines.append(data_line)

    text_all = "\n".join(text_lines)

    if gap_filled and os.path.split(csv_file)[1].find("gap_filled") == -1:
        base, ext = os.path.splitext(csv_file)
        csv_file = base + "_gap_filled" + ext

    # Save the CSV.
    with open(csv_file, 'w') as f:
        f.write(text_all)

    if verbose:
        print(csv_file, "written.")


if __name__ == "__main__":

    for gap_filled in (True,):
        save_climatologies_as_CSV(gap_filled=gap_filled)
        save_daily_melt_numbers_to_csv(gap_filled=gap_filled)
        # create_baseline_climatology_tif(gap_filled=gap_filled)
        # create_annual_melt_sum_tif(year="all", gap_filled=gap_filled)
        # for year in range(1979,2020):
        #     create_annual_melt_anomaly_tif(year, gap_filled=gap_filled)



    # compute_daily_climatology_pixel_averages()
    # compute_daily_sum_pixel_averages()
    # create_partial_year_melt_anomaly_tif()
