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

from ssmi_simple_read import read_SSMI_data

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

def get_hemisphere_and_resolution_from_ssmi_filename(fname):
    '''From the file specs on https://nsidc.org/data/nsidc-0001, get the resolution
    from the filename. Will be 12.5 or 25 km.'''
    fbase = os.path.splitext(os.path.split(fname)[1])[0]

    SSMI_REGEX =  r"(?<=\Atb_f\d{2}_\d{8}_v\d_)[ns]\d{2}(?=[vh])"

    matches = re.search(fbase, SSMI_REGEX)
    if matches is None:
        return None, None

    match = matches.group(0)
    hemisphere = match[0].upper()

    frequency = int(match[1:2])

    resolution = {19:25.0,
                  22:25.0,
                  37:25.0,
                  85:12.5,
                  91:12.5}[frequency]

    return hemisphere, resolution

def output_bin_to_gtif(bin_file,
                       gtif_file=None,
                       resolution=None,
                       hemisphere=None,
                       verbose=True,
                       nodata=0,
                       return_type=float):
    '''Read an NSIDC SSMI .bin file and output to a geo-referenced .tif file.

    The hemisphere and spatial resolution are acquired from the filename. File
    names should be kept as downloaded from the NSIDC. Changed file names do not
    guarantee good outputs.'''

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

    assert resolution in (6.25, 12.5, 25.0)
    assert hemisphere in ("N", "S")

    gridsize_dict = {(6.25, "N"):GRIDSIZE_6_25_N,
                     (6.25, "S"):GRIDSIZE_6_25_S,
                     (12.5, "N"):GRIDSIZE_12_5_N,
                     (12.5, "S"):GRIDSIZE_12_5_S,
                     (25.0, "N"):GRIDSIZE_25_N,
                     (25.0, "S"):GRIDSIZE_25_S}

    # Read in the array
    array = read_SSMI_data(bin_file,
                           grid_shape = gridsize_dict[(resolution, hemisphere)],
                           return_type=return_type)

    # Export the file.
    output_gtif(array,
                gtif_file,
                resolution=resolution,
                hemisphere=hemisphere,
                nodata=nodata,
                verbose=verbose)

    return

def get_nsidc_geotransform(hemisphere, resolution):
    '''Given the hemisphere and the resolution of the dataset, return the
    6-number GeoTiff 'geotransform' tuple.'''
    # Must multiply km resolution by 1000 to get meters, for the projection.
    if hemisphere.strip().upper() == "N":
        UL_X, UL_Y = NSIDC_N_GRID_UPPER_LEFT_KM * 1000
    elif hemisphere.strip().upper() == "S":
        UL_X, UL_Y = NSIDC_S_GRID_UPPER_LEFT_KM * 1000
    else:
        raise ValueError("Unknown hemisphere: '{0}'".format(hemisphere))

    return (UL_X, resolution*1000, 0, UL_Y, 0, -resolution*1000)


def output_gtif(array, gtif_file, resolution=25, hemisphere="S", nodata=0, verbose=True):
    '''Take an array, output to a geotiff in the NSIDC resolution specified.
    Defaults to 25 km resolution, southern hemisphere.'''
    geotransform = get_nsidc_geotransform(hemisphere=hemisphere,
                                          resolution=resolution)

    driver = gdal.GetDriverByName("GTiff")
    if array.dtype == int:
        datatype = gdal.GDT_UInt16
    elif array.dtype == float:
        datatype = gdal.GDT_Float32
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
    array_wo_nodata = array[array != nodata]

    ds = driver.Create(gtif_file, array.shape[1], array.shape[0], 1, datatype)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.SetStatistics(numpy.min(array_wo_nodata),
                       numpy.max(array_wo_nodata),
                       numpy.mean(array_wo_nodata),
                       numpy.std(array_wo_nodata))
    ds.FlushCache()
    ds = None

    if verbose:
        print(gtif_file, "written.")

    return


def read_and_parse_args():
    parser = argparse.ArgumentParser(description="Outputs a GTIFF from an NSIDC SMMI Polar Stereo Brightness Temperature data file.")
    parser.add_argument("src", type=str, help="Source file (.bin)")
    parser.add_argument("-dest", type=str, default="", help="Destination file (.tif)")
    parser.add_argument("--resolution", "-r", type=float, default=None, help="Resolution (km): 6.25, 12.5, or 25. If omitted, it is interpreted from the file name. If cannot be interpreted, defaults to 25 km.")
    parser.add_argument("--hemisphere", type=str, default=None, help="Hemistphere: N or S. If omitted, it is interpreted from the file name. If cannot be interpreted, defaults to 'N'.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Increase output verbosity.")
    parser.add_argument("--nodata", "-nd", type=int, default=0, help="Nodata value. (Default: 0)")
    parser.add_argument("--output_type", "-ot", default="float", help="Output data type: 'int' or 'float'. Default 'float'.")

    return parser.parse_args()

if __name__ == "__main__":
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

    output_bin_to_gtif(args.src,
                       args.dest,
                       resolution = resolution,
                       hemisphere = hemisphere,
                       nodata = int(args.nodata),
                       return_type = out_type,
                       verbose = args.verbose)