""" generate_daily_melt_file.py -- Reads a daily NSIDC Tb file, the annual
Tb cutoff file, an ice mask file, and generates an output integer daily melt file
(flat binary) in the following format:
    -1 : Not within ice mask
     0 : Tb value missing that day
    +1 : No melt (Tb < threshold)
    +2 : Melt (Tb >= threshold)

Can output .bin, .tif, or both (see command-line options).
Run with no parameters, or with the -h parameter, to see command-line options.
"""

import argparse
import datetime
import math
import os
import re
import warnings
from pathlib import Path

import numpy
import xarray

from antarctica_today import tb_file_data
from antarctica_today.melt_array_picklefile import get_ice_mask_array
from antarctica_today.read_NSIDC_bin_file import read_NSIDC_bin_file
from antarctica_today.read_NSIDC_nc_file import read_NSIDC_nc_file
from antarctica_today.write_flat_binary import write_array_to_binary


def generate_new_daily_melt_files(
    start_date="2021-10-01",
    end_date=None,
    overwrite=True,
    warn_if_missing_files=True,
):
    """Look through the .bin melt files, and create new ones.

    This function assumes the necessary .bin Tb files from NSIDC are downloaded.
    If not, go to "nsidc_download_Tb_data.py" and update there first.
    """
    start_dt = datetime.datetime(
        year=int(start_date[0:4]), month=int(start_date[5:7]), day=int(start_date[8:10])
    )
    now = datetime.datetime.today()
    if end_date is None:
        end_dt = datetime.datetime(year=now.year, month=now.month, day=now.day)
    else:
        end_dt = datetime.datetime(
            year=int(end_date[0:4]), month=int(end_date[5:7]), day=int(end_date[8:10])
        )

    # Get a list of .bin files already in the directory.
    existing_melt_files = [
        fname
        for fname in os.listdir(tb_file_data.model_results_dir)
        if os.path.splitext(fname)[1].lower() == ".bin"
    ]

    for day_offset in range((end_dt - start_dt).days + 1):
        dt = start_dt + datetime.timedelta(days=day_offset)

        # Only overwrite if a .bin melt file doesn't already exist there.
        skip_this_file = False
        for meltfile in existing_melt_files:
            match = re.search(
                dt.strftime("(?<=antarctica_melt_)%Y%m%d(?=_S3B)"), meltfile
            )
            if match != None:
                if overwrite:
                    os.remove(os.path.join(tb_file_data.model_results_dir, meltfile))
                else:
                    skip_this_file = True

        if skip_this_file:
            continue

        # Find files in the NSIDC-0080 directory that match this date.
        nsidc_dir = tb_file_data.NSIDC_0080_file_dir
        # First fine all the .bin files that match that date stamp in the string
        nsidc_fps: list[Path] = [
            fp
            for fp in nsidc_dir.iterdir()
            if (fp.name.find(dt.strftime("%Y%m%d")) > -1 and fp.suffix.lower() == ".nc")
        ]

        # There shouldn't be more than one file on that date.
        assert len(nsidc_fps) <= 1

        # Make sure there's at least one of each file (i.e. exactly one). If not, just skip & continue
        if len(nsidc_fps) == 0:
            if warn_if_missing_files:
                warnings.warn(
                    "Warning: At least one NSIDC Tb file on date '"
                    + dt.strftime("%Y%m%d")
                    + "' is missing. Skipping that date."
                )
            continue

        threshold_file = get_correct_threshold_file(dt)
        if threshold_file is None:
            continue

        outfile_name = dt.strftime("antarctica_melt_%Y%m%d_S3B_") + now.strftime(
            "%Y%m%d.bin"
        )
        outfile_path = tb_file_data.model_results_dir / outfile_name
        create_daily_melt_file(
            nsidc_fps[0],
            threshold_file,
            outfile_path,
        )


def create_daily_melt_file(
    nsidc_0080_fp: Path,
    threshold_file,
    output_bin_filename,
    output_gtif_filename=None,
    Tb_nodata_value=-999,
    verbose=True,
):
    """Read input files and generate a daily melt file. Primary function."""
    output_array = read_files_and_generate_melt_array(
        nsidc_0080_fp,
        threshold_file,
        Tb_nodata_value=Tb_nodata_value,
    )

    # Write the output .bin file
    # write_flat_binary.write_array_to_binary(
    write_array_to_binary(
        output_array, output_bin_filename, numbytes=2, signed=True, verbose=verbose
    )

    # Write the output.tif file, if called for
    if output_gtif_filename != None:
        write_NSIDC_bin_to_gtif.output_gtif(
            output_array,
            output_gtif_filename,
            resolution=25,
            hemisphere="S",
            nodata=None,
            verbose=verbose,
        )

    return output_array


def get_melt_year_of_current_date(
    dt_object, melt_doy_start=(10, 1), melt_doy_end=(4, 30)
):
    """For a given datetime object, return the melt year number that corresponds to that date.
    If the data falls outside of the melt year, return None.
    """
    # Find out which year this particular date should correspond with.
    year, month, day = dt_object.year, dt_object.month, dt_object.day
    doy = (month, day)
    # In Antarctica, since the melt season spans the calendar new year.
    if melt_doy_start > melt_doy_end:
        # If this date is outside the melt season, just return None
        if doy < melt_doy_start and doy > melt_doy_end:
            return None
        # If it's before the new year, return the current year
        if doy >= melt_doy_start:
            return year
        # If it's after the new year, then return last yeaer (which was the start of the melt season)
        elif doy <= melt_doy_end:
            return year - 1
        else:
            raise RuntimeError(
                "Should never get to this point. Something went wrong in the logic."
            )

    # If the melt year ends in the same calendar year as the start, the logic is a bit simpler.
    else:
        # If it's outside the melt season, return None
        if doy < melt_doy_start or doy > melt_doy_end:
            return None
        # If it's inside the melt season, return the current year.
        else:
            return year


def get_correct_threshold_file(
    dt_object,
    melt_doy_start=(10, 1),
    melt_doy_end=(4, 30),
    thresholds_dir=tb_file_data.threshold_file_dir,
):
    """For a given datetime object, return the threshold file that corresponds to that date.
    If the data falls outside of the melt year,
    or the threshold file doesn't exist in the directory, quietly return None.
    """
    # Get the list of all the threshold files.
    threshold_files = os.listdir(thresholds_dir)
    # Find out which year this particular date should correspond with.
    year = get_melt_year_of_current_date(
        dt_object, melt_doy_start=melt_doy_start, melt_doy_end=melt_doy_end
    )
    if year is None:
        return None
    year_str = str(year)

    # If we find a threshold file with the selected year in it, return its path.
    # This makes the assumption that the threshold file names only have one spot with a given year in the filename.
    # If this is not the case, this code must change.
    for tf in threshold_files:
        if year_str in tf:
            return os.path.join(thresholds_dir, tf)

    return None


def read_files_and_generate_melt_array(
    nsidc_0080_fp: Path, threshold_file, Tb_nodata_value=-999
):
    """Generate a daily melt value array from the three flat-binary files."""
    # TODO: F18 correct?
    nsidc_0080 = read_NSIDC_nc_file(nsidc_0080_fp)
    threshold_array = read_NSIDC_bin_file(
        threshold_file, return_type=float, multiplier=0.1
    )
    ice_mask_array = get_ice_mask_array()

    return create_daily_melt_array(
        **nsidc_0080,
        threshold_array=threshold_array,
        ice_mask_array=ice_mask_array,
        Tb_nodata_value=Tb_nodata_value,
    )


def create_daily_melt_array(
    *,
    Tb_array_37h,
    Tb_array_37v,
    Tb_array_19v,
    threshold_array,
    ice_mask_array,
    Tb_nodata_value=-999
):
    """Create an NxM array of daily melt values.

    This function uses the parameters set out for Antarctica Today to derive melt
    from passive microwave brightness temperatures using the 37 GHz h channel, with
    a filtering step using the 19 GHz h.

    It is considered melt if:
        - The 37v GHz Tb < 19v GHz Tb and 37h GHz Tb > annual threshold,
        OR
        - The 37v GHz Tb > 19v GHz Tb and 37h GHz Tb > (annual threshold + 10K)

    Parameters
    ----------
    - Tb_array: NumPy array of Tb values
    - threshold_array: NumPy array of annual Tb threshold values
    - ice_mask_array: NumPy ice mask array
    - Tb_nodata_value: Nodata value in the Tb array (default 0)

    Return value
    ------------
    A numpy array of daily melt values:
        -1 : Not within ice mask
         0 : Tb value missing that day
        +1 : No melt (Tb < threshold)
        +2 : Melt (Tb >= threshold)
    """
    # All these arrays should be the same size and shape
    assert (
        Tb_array_37h.shape
        == Tb_array_37v.shape
        == Tb_array_19v.shape
        == threshold_array.shape
        == ice_mask_array.shape
    )

    # Create empty output array. Use -999 as a temporary empty_value to ensure all cells get assigned something.
    output_array = numpy.zeros(Tb_array_37h.shape, dtype=numpy.int16) - 999

    # No melt if it doesn't exceed the threshold at all.
    output_array[Tb_array_37h < threshold_array] = 1
    # Melt if the (Tb_19v - Tb_37v) >= 0 and (Tb_37h > threshold)
    output_array[
        ((Tb_array_19v - Tb_array_37v) >= 0) & (Tb_array_37h >= threshold_array)
    ] = 2
    # Melt if the (Tb_19v - Tb_37v) < 0 and (Tb_37h >= threshold + 10k)
    output_array[
        ((Tb_array_19v - Tb_array_37v) < 0) & (Tb_array_37h >= (threshold_array + 10))
    ] = 2
    # No melt if (Tb_19h - Tb_37h) < 0 and (Tb_38 < threshold + 10k)
    output_array[
        ((Tb_array_19v - Tb_array_37v) < 0) & (Tb_array_37h < (threshold_array + 10))
    ] = 1

    # Mark all "nodata" as no data.

    Tb_array_37h[numpy.isnan(Tb_array_37h)] = -999
    Tb_array_37v[numpy.isnan(Tb_array_37v)] = -999
    Tb_array_19v[numpy.isnan(Tb_array_19v)] = -999

    output_array[
        (Tb_array_37h == Tb_nodata_value)
        | (Tb_array_37v == Tb_nodata_value)
        | (Tb_array_19v == Tb_nodata_value)
    ] = 0
    # Everything outside the mask is -1
    output_array[ice_mask_array == 0] = -1

    # The above should have filled in all the values. Just make sure there are no -999's left here.
    assert numpy.count_nonzero(output_array == -999) == 0

    return output_array


# TODO: If we want to expose creating one daily melt file as a CLI, we should also
# expose creating them all. Currently the code to create them all isn't exposed
# anywhere!
# def read_and_parse_args():
#     """Read and parse the command-line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Generates a single daily melt file from NSIDC Tb files."
#     )
#     parser.add_argument(
#         "Tb_file_37h", type=str, help="A daily NSIDC Polar-stereo Tb files (.bin)"
#     )
#     parser.add_argument(
#         "Tb_file_37v", type=str, help="A daily NSIDC Polar-stereo Tb files (.bin)"
#     )
#     parser.add_argument(
#         "Tb_file_19v", type=str, help="A daily NSIDC Polar-stereo Tb files (.bin)"
#     )
#     parser.add_argument(
#         "threshold_file", type=str, help="A file of Tb threshold values (.bin)"
#     )
#     parser.add_argument("output_file", type=str, help="Integer output file (.bin)")
#     parser.add_argument(
#         "--ouput_gtif",
#         action="store_true",
#         default=False,
#         help="Output a GeoTiff (.tif) in addition to the flat binary.",
#     )
#     parser.add_argument(
#         "--verbose",
#         "-v",
#         action="store_true",
#         default=False,
#         help="Increase output verbosity.",
#     )
#
#     return parser.parse_args()

# if __name__ == "__main__":
# args = read_and_parse_args()

# # def create_daily_melt_file(tb_file_37h,
# #                            tb_file_37v,
# #                            tb_file_19v,
# #                            threshold_file,
# #                            output_bin_filename,
# #                            output_gtif_filename = None,
# #                            Tb_nodata_value = 0,
# #                            verbose = True):

# if args.output_gtif:
#     gtif_name = os.path.splitext(args.output_file)[0] + ".tif"
# else:
#     gtif_name = None

# create_daily_melt_file(
#     args.Tb_file_37h,
#     args.Tb_file_37v,
#     args.Tb_file_19v,
#     args.threshold_file,
#     args.output_file,
#     output_gtif_filename=gtif_name,
#     verbose=args.verbose,
# )

if __name__ == "__main__":
    generate_new_daily_melt_files(overwrite=False)
