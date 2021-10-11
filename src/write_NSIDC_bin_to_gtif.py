# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:21:04 2020

@author: mmacferrin
"""
import numpy
import argparse
import re
import os
from osgeo import osr, gdal

from read_NSIDC_bin_file import read_NSIDC_bin_file

# See https://nsidc.org/data/polar-stereo/ps_grids.html for documentation on
# these polar stereo grids
# Upper-left corners of the grids, in km, in x,y
NSIDC_S_GRID_UPPER_LEFT_KM = numpy.array((-3950,4350), dtype=numpy.int)
NSIDC_N_GRID_UPPER_LEFT_KM = numpy.array((-3850,5850), dtype=numpy.int)
# Pixel dimentions of the respective grids, in (y,x) --> (rows, cols)
GRIDSIZE_25_N = numpy.array(((5850+5350)/25, (3750+3850)/25), dtype=numpy.long) # (448, 304)
GRIDSIZE_25_S = numpy.array(((4350+3950)/25, (3950+3950)/25), dtype=numpy.long) # (332, 316)
GRIDSIZE_12_5_N = GRIDSIZE_25_N * 2 # (896, 608)
GRIDSIZE_12_5_S = GRIDSIZE_25_S * 2 # (664, 632)
GRIDSIZE_6_25_N = GRIDSIZE_25_N * 4
GRIDSIZE_6_25_S = GRIDSIZE_25_S * 4
# EPSG reference numbers for each of the grids.
EPSG_N = 3411
SPATIAL_REFERENCE_N = osr.SpatialReference()
SPATIAL_REFERENCE_N.ImportFromEPSG(EPSG_N)

EPSG_S = 3412
SPATIAL_REFERENCE_S = osr.SpatialReference()
SPATIAL_REFERENCE_S.ImportFromEPSG(EPSG_S)

def retrieve_ssmi_grid_coords(N_or_S="S", gridsize_km=25):
    """Return two arrays, for "grid_x" and "grid_y" corrdinates of the array."""
    if N_or_S.strip().upper() == "S":
        hemisphere="S"
        UL_corner=NSIDC_S_GRID_UPPER_LEFT_KM
    elif N_or_S.strip().upper() == "N":
        hemisphere="N"
        UL_corner=NSIDC_N_GRID_UPPER_LEFT_KM
    else:
        raise ValueError("Uknown hemisphere " + str(N_or_S))

    assert gridsize_km in (25,12.5,6.25)
    if hemisphere=="N":
        if gridsize_km==25:
            gridsize_yx = GRIDSIZE_25_N
        elif gridsize_km==12.5:
            gridsize_yx = GRIDSIZE_12_5_N
        elif gridsize_km==6.25:
            gridsize_yx = GRIDSIZE_6_25_N
    if hemisphere=="S":
        if gridsize_km==25:
            gridsize_yx = GRIDSIZE_25_S
        elif gridsize_km==12.5:
            gridsize_yx = GRIDSIZE_12_5_S
        elif gridsize_km==6.25:
            gridsize_yx = GRIDSIZE_6_25_S

    x_vector = numpy.arange(UL_corner[0], UL_corner[0]+(gridsize_km*gridsize_yx[1]), step=gridsize_km)
    y_vector = numpy.arange(UL_corner[1], UL_corner[1]+(-gridsize_km*gridsize_yx[0]), step=-gridsize_km)
    return x_vector, y_vector

def get_hemisphere_and_resolution_from_ssmi_filename(fname):
    """Get the resolution from the filename.

    From the file specs on https://nsidc.org/data/nsidc-0001
    Will be 12.5 or 25 km.
    """
    fbase = os.path.splitext(os.path.split(fname)[1])[0]

    SSMI_REGEX =  r"(?<=\Atb_f\d{2}_\d{8}_v\d_)[ns]\d{2}(?=[vh])"

    matches = re.search(fbase, SSMI_REGEX)
    if matches is None:
        return None, None

    match = matches.group(0)
    hemisphere = match[0].upper()

    frequency = int(match[1:2])

    # The resolutions for each frequency in the NSIDC data products.
    # Dictionary is "frequency:resolutin" key:value pair.
    resolution = {19:25.0,
                  22:25.0,
                  37:25.0,
                  85:12.5,
                  91:12.5}[frequency]

    return hemisphere, resolution

def output_bin_to_gtif(bin_file,
                       gtif_file=None,
                       element_size=2,
                       header_size=0,
                       resolution=None,
                       hemisphere=None,
                       verbose=True,
                       nodata=0,
                       signed=False,
                       multiplier="auto",
                       return_type=float):
    """Read an NSIDC SSMI .bin file and output to a geo-referenced .tif file.

    The hemisphere and spatial resolution are acquired from the filename. File
    names should be kept as downloaded from the NSIDC. Changed file names do not
    guarantee good outputs.

    bin_file = Name of the flat-binary data file to read.

    gtif_file = Name of the geotiff to produce.
                If None, it uses the same filname as "bin_file" with the
                file extension swapped with ".tif". NOTE: If a .tif file is given
                for "bin_file", this will overwrite the file. (And probably break
                anyway since a .tif is not a flat-binary file.)

    resolution = Floating-point grid resolution, in km.
                 Accepted values are: 25.0, 12.5, 6.25
                 If None, the resolution is derived from the nsidc-0001 filename.

    hemisphere = "N" or "S"
                 If None, the hemisphere is derived from the nsidc-0001 filename.

    verbose = Verbosity of the output. False will run this silently. True will
              produce feedback to stdout. (default True)

    nodata = Nodata value to put in the geotiff. Defaults to 0.0

    return_type = The data type of the geotiff raster band. Defaults to float.
    """
    if (gtif_file is None) or (len(gtif_file.strip().upper()) == 0):
        gtif_file = os.path.splitext(bin_file)[0] + ".tif"

    if resolution is None or hemisphere is None:
        # Get hemisphere & resolution from file name
        hemisphere_from_fname, resolution_from_fname = \
            get_hemisphere_and_resolution_from_ssmi_filename(bin_file)
        # Only replace values if not explicitly given.
        if resolution is None:
            resolution = resolution_from_fname
            if resolution is None:
                resolution = 25.0
        if hemisphere is None:
            hemisphere = hemisphere_from_fname
            if hemisphere is None:
                hemisphere = "S"

    resolution = float(resolution)
    assert resolution in (6.25, 12.5, 25.0)
    assert hemisphere in ("N", "S")

    gridsize_dict = {(6.25, "N"):GRIDSIZE_6_25_N,
                     (6.25, "S"):GRIDSIZE_6_25_S,
                     (12.5, "N"):GRIDSIZE_12_5_N,
                     (12.5, "S"):GRIDSIZE_12_5_S,
                     (25.0, "N"):GRIDSIZE_25_N,
                     (25.0, "S"):GRIDSIZE_25_S}

    if multiplier.strip().lower() =="auto":
        multiplier= 1 if (return_type == int) else 0.1

    # Read in the array
    array = read_NSIDC_bin_file(bin_file,
                                grid_shape = gridsize_dict[(resolution, hemisphere)],
                                header_size=header_size,
                                element_size=element_size,
                                return_type=return_type,
                                signed=signed,
                                multiplier=multiplier)

    # Export the file.
    output_gtif(array,
                gtif_file,
                resolution=resolution,
                hemisphere=hemisphere,
                nodata=nodata,
                verbose=verbose)

    return

def get_nsidc_geotransform(hemisphere, resolution):
    """Given the hemisphere and the resolution of the dataset, return the 6-number GeoTiff 'geotransform' tuple."""
    # Must multiply km resolution by 1000 to get meters, for the projection.
    if hemisphere.strip().upper() == "N":
        UL_X, UL_Y = NSIDC_N_GRID_UPPER_LEFT_KM * 1000
    elif hemisphere.strip().upper() == "S":
        UL_X, UL_Y = NSIDC_S_GRID_UPPER_LEFT_KM * 1000
    else:
        raise ValueError("Unknown hemisphere: '{0}'".format(hemisphere))

    return (UL_X, resolution*1000, 0, UL_Y, 0, -resolution*1000)


def output_gtif(array, gtif_file, resolution=25, hemisphere="S", nodata=0, verbose=True):
    """Take an array, output to a geotiff in the NSIDC resolution specified.

    Defaults to 25 km resolution, southern hemisphere.
    """
    geotransform = get_nsidc_geotransform(hemisphere=hemisphere,
                                          resolution=resolution)

    driver = gdal.GetDriverByName("GTiff")
    if array.dtype in (numpy.int8, numpy.int16, numpy.int32, numpy.int64):
        if array.dtype in (numpy.int8, numpy.int16):
            datatype = gdal.GDT_Int16
        elif array.dtype in (numpy.int32, numpy.int64):
            datatype = gdal.GDT_Int32

    elif array.dtype in (numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint32):
        if array.dtype in (numpy.uint8, numpy.uint16):
            datatype = gdal.GDT_UInt16
        elif array.dtype in (numpy.uint32, numpy.uint64):
            datatype = gdal.GDT_UInt32

    elif array.dtype == numpy.float32:
        datatype = gdal.GDT_Float32

    elif array.dtype == numpy.float64:
        datatype = gdal.GDT_Float64

    else:
        raise TypeError("Unhandled data type {0}. Please use int or float.".format(str(array.dtype)))

    hemisphere_upper = hemisphere.strip().upper()
    if hemisphere_upper == "S":
        projection = SPATIAL_REFERENCE_S
    elif hemisphere_upper == "N":
        projection = SPATIAL_REFERENCE_N
    else:
        raise ValueError("Unknown hemisphere", hemisphere)

    # Calculate statistics
    if nodata != None:
        array_wo_nodata = array[array != nodata]
    else:
        array_wo_nodata = array

    ds = driver.Create(gtif_file, array.shape[1], array.shape[0], 1, datatype)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(array)

    if nodata != None:
        band.SetNoDataValue(nodata)

    if len(array_wo_nodata) > 0:
        # Only set statistics if this isn't an empty array.
        band.SetStatistics(float(numpy.min(array_wo_nodata)),
                           float(numpy.max(array_wo_nodata)),
                           float(numpy.mean(array_wo_nodata)),
                           float(numpy.std(array_wo_nodata)))
    else:
        band.SetStatistics(numpy.NaN,
                           numpy.NaN,
                           numpy.NaN,
                           numpy.NaN)

    ds.FlushCache()
    ds = None

    if verbose:
        print(gtif_file, "written.")

    return


def read_and_parse_args():
    """Read and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Outputs a GTIFF from an NSDIC flat binary (.bin) data file.\n(Since NSIDC still hasn't figured out that not everyone wants to use IDL.)")
    parser.add_argument("src", type=str, help="Source file (.bin)")
    parser.add_argument("-dest", type=str, default="", help="Destination file (.tif). Default: Write the same filename in the same location with a .tif extension rather than .bin.")
    parser.add_argument("-resolution", "-r", type=float, default=None, help="Resolution (km): 6.25, 12.5, or 25. If omitted, it is interpreted from the file name. If cannot be interpreted, defaults to 25 km. Check your NSIDC data source documentation.")
    parser.add_argument("-hemisphere", type=str, default=None, help="Hemisphere: N or S. If omitted, it is interpreted from the file name. If cannot be interpreted, defaults to 'N'.")
    parser.add_argument("-nodata", "-nd", type=int, default=0, help="Nodata value. (Default: 0). IMPORTANT: If you want a nodata value other than 0, you should specify it. If you don't want to use a nodata value, then specify a nonsense value that is not contained in the data (such as -99999).")
    parser.add_argument("-header_size", "-hs", type=int, default=0, help="Size of .bin file header (in bytes.) (Default: 0)")
    parser.add_argument("-element_size", "-es", type=int, default=2, help="Size of each numerical .bin data element, in bytes. (Default: 2)")
    parser.add_argument("-output_type", "-ot", default="float", help="Output data type: 'int' or 'float'. Default 'float'.")
    parser.add_argument("-multiplier","-m", type=str, default="auto", help="Use a multiplier. With 'auto', defaults to 1 for integers (no mod) and 0.1 for floating-point. If you want to use a different multiplier, put the number here.")
    parser.add_argument("--signed", "-s", action="store_true", default=False, help="Read bin as signed data. Default to unsigned.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Increase output verbosity.")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse the command-line arguments.
    args = read_and_parse_args()
    if args.dest == "":
        dest = None
    else:
        dest = args.dest

    # Parse the resolution argument
    if args.resolution == None:
        resolution = None
    else:
        try:
            if float(args.resolution) not in (6.25, 12.5, 25.0):
                raise ValueError("Unknown resolution: {0}".format(args.resolution))
            else:
                resolution = float(args.resolution)
        except ValueError:
            raise ValueError("Unknown resolution: {0}".format(args.resolution))

    assert resolution in (None, 6.25, 12.5, 25.0)

    # Parse the hemisphere argument
    if args.hemisphere == None:
        hemisphere = None
    else:
        try:
            if args.hemisphere.strip().upper() not in ("N", "S"):
                raise ValueError("Unknown hemisphere: {0}".format(args.resolution))
            else:
                hemisphere = args.hemisphere.strip().upper()
        except AttributeError:
            raise ValueError("Unknown hemisphere: {0}".format(args.resolution))

    assert hemisphere in (None, "N", "S")

    if args.output_type.lower() in ("float", "f"):
        out_type = float
    elif args.output_type.lower() in ("int", "i", "d"):
        out_type = int
    else:
        raise ValueError("Uknown output_type (can be: 'int','i','d','float', or 'f'):", str(args.output_type))

    if args.multiplier.lower().strip() != "auto":
        multiplier = float(args.multiplier)
    else:
        multiplier = args.multiplier

    output_bin_to_gtif(args.src,
                       args.dest,
                       header_size = args.header_size,
                       element_size = args.element_size,
                       resolution = resolution,
                       hemisphere = hemisphere,
                       nodata = int(args.nodata),
                       return_type = out_type,
                       multiplier = multiplier,
                       verbose = args.verbose)