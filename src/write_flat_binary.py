# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:38:32 2020

@author: mmacferrin

write_flat_binary.py -- code to output a .tif or numpy array to a flat-binary file.

"""

import numpy
import argparse
from osgeo import gdal
import os

def write_array_to_binary(array,
                          bin_filename,
                          numbytes=2,
                          multiplier=1,
                          byteorder="little",
                          signed=False,
                          verbose=True):

    if int(numbytes) not in (1,2,4,8):
        raise ValueError("Numbytes must be one of 1,2,4,8.")

    # Get the byte order.
    byteorder = byteorder.strip().lower()
    if byteorder not in ("little", "big", "l", "b"):
        raise ValueError("byteorder must be 'little', 'L', 'big', or 'B'.")
    elif byteorder == "l":
        byteorder = "little"
    elif byteorder == "b":
        byteorder = "big"

    # Open the output file name.
    f = open(bin_filename, 'wb')

    # Convert the number of bytes into the correct numpy array datatype.
    if signed:
        n_dtype = {1:numpy.int8,
                   2:numpy.int16,
                   4:numpy.int32,
                   8:numpy.int64 }[int(numbytes)]
    else:
        n_dtype = {1:numpy.uint8,
                   2:numpy.uint16,
                   4:numpy.uint32,
                   8:numpy.uint64 }[int(numbytes)]

    # Converte the array into the appropriate data type, and multiply by the multiplier
    out_array = numpy.array(array * multiplier, dtype=n_dtype)

    # Flatten the array.
    out_array = out_array.flatten()

    for value in out_array:
        f.write(int.to_bytes(int(value), length=numbytes, byteorder=byteorder, signed=signed))

    f.close()

    if verbose:
        print(os.path.split(bin_filename)[-1], "written.")

    return bin_filename


def write_gtif_to_binary(gtif_filename,
                         bin_filename=None,
                         rasterband=1,
                         numbytes=2,
                         multiplier=1,
                         byteorder="little",
                         signed=False,
                         verbose=True):

    rasterband = int(rasterband)
    if rasterband < 1:
        raise ValueError("Raster band must be an integer greater than or equal to 1.")

    if verbose:
        print("Reading",os.path.split(gtif_filename)[-1])


    ds = gdal.Open(gtif_filename, gdal.GA_ReadOnly)
    if ds is None:
        raise Exception("{0} not read correctly by GDAL.".format(gtif_filename))

    if rasterband > ds.RasterCount:
        raise ValueError("File {0} contains {1} raster bands. Cannot read raster band {2}.".format(
                         os.path.split(gtif_filename)[-1],
                         ds.RasterCount,
                         rasterband)
                        )

    band = ds.GetRasterBand(rasterband)
    array = band.ReadAsArray()

    if bin_filename is None or len(bin_filename.strip()) == 0:
        bin_filename = os.path.splitext(gtif_filename)[0] + '.bin'

    return write_array_to_binary(array, bin_filename=bin_filename,
                                        numbytes=numbytes,
                                        multiplier=multiplier,
                                        byteorder=byteorder,
                                        signed=signed,
                                        verbose=verbose)



def read_and_parse_args():
    parser = argparse.ArgumentParser(description="Outputs a flat binary integer files (in the style of NSIDC SMMI Polar Stereo Brightness Temperature data) from a GeoTiff.")
    parser.add_argument("gtif", type=str, help="Source file (.tif)")
    parser.add_argument("-output", "-o", type=str, default="", help="Destination file (.bin)")
    parser.add_argument("-band", "-b", type=int, default=1, help="Raster band number in TIF file. Any integer from 1 to the number of available rastrer bands. Defaults to 1.")
    parser.add_argument("-numbytes", "-nb", type=int, default=2, help="Number of output bytes. Can be 1,2,4,8. (Default: 2)")
    parser.add_argument("-multiplier", "-m", type=int, default=1, help="Multiplier of .tif values.\n" + \
                        "Useful for converting floating point .tif to integer .bin.\n" + \
                        "For instance, turning 273.2 to 2732 would use a multiplier of 10. Default 1.")
    parser.add_argument("-byteorder", "-bo", type=str, default="little", help="Byte-order of the integer values to write.\n" + \
                        "Values 'little' or 'big'. Default: 'little'.")
    parser.add_argument("--signed", "-s", action="store_true", default=False, help="Signed data. Defaults to unsigned. Results will be the same if no negative values are in the array.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Increase output verbosity.")

    return parser.parse_args()

if __name__ == "__main__":
    args = read_and_parse_args()

    write_gtif_to_binary(args.gtif,
                         None if (args.output.strip()=="") else args.output.strip(),
                         numbytes=args.numbytes,
                         multiplier=args.multiplier,
                         byteorder=args.byteorder,
                         verbose=args.verbose)
