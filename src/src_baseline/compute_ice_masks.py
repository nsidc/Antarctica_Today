# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:12:42 2020

@author: mmacferrin
"""

default_input_coastline_tif = r'F:/Research/DATA/Antarctica_Today/baseline_datasets/ice mask/ADD_Coastlines_1km.tif'
default_input_rock_outcrop_tif = r'F:/Research/DATA/Antarctica_Today/baseline_datasets/ice mask/Rock_outcrop_LandSat8_1km.tif'
default_output_dir = r'F:/Research/DATA/Antarctica_Today/baseline_datasets/'

from osgeo import gdal, osr
import os
import numpy
from write_flat_binary import write_array_to_binary

def compute_ice_masks(input_coastline_tif       = default_input_coastline_tif,
                      input_bedrock_outcrop_tif = default_input_rock_outcrop_tif,
                      output_dir                = default_output_dir):
    '''Take a high-res ice boundary tif file, and a bedrock outcrop tif file, and
    compute ice mask files (both .tif and .bin).

    This function is very specific for this purpose, and not generalized. The input
    TIFs are 250 m resolution at the exact same grid boundaries as the output SSMI
    NSIDC South Polar Stereo brightness temperature files. 1 pixel in SSMI is 100
    pixels in each input tif, perfectly aligned.

    Output 4 sets of files, in both .tif and .bin (8 files total):
        fraction_ocean (0 to 1)
        fraction_rock (0 to 1)
        fraction_ice (0 to 1)
        binary_ice (0 or 1, no other values)

        binary_ice is defined as 1 if fraction_ice is >=0.50, 0 otherwise

        For .tif files, save as floating point for fractions, 8-bit int for binary_ice.
        For .bin files, all are saved a flat 16-bit integer. For fractions, it's
        10x the % of each value. For instance, 0310 refers to 0.31, or 31.0%'
    '''

    # Read in the geotiffs

    # Coastline geotiff
    print("Reading", input_coastline_tif)
    ds_coastline = gdal.Open(input_coastline_tif, gdal.GA_ReadOnly)
    if ds_coastline is None:
        raise FileNotFoundError("Gdal could not read input file '{0}'".format(input_coastline_tif))

    gt_coastline = ds_coastline.GetGeoTransform()
    band_coastline = ds_coastline.GetRasterBand(1)
    array_coastline = ds_coastline.ReadAsArray()
    ndv_coastline = band_coastline.GetNoDataValue()

    # Rock geotiff
    print("Reading", input_bedrock_outcrop_tif)
    ds_rock = gdal.Open(input_bedrock_outcrop_tif, gdal.GA_ReadOnly)
    if ds_rock is None:
        raise FileNotFoundError("Gdal could not read input file '{0}'".format(input_bedrock_outcrop_tif))

    gt_rock = ds_rock.GetGeoTransform()
    band_rock = ds_rock.GetRasterBand(1)
    array_rock = ds_rock.ReadAsArray()
    ndv_rock = band_rock.GetNoDataValue()

    # Sanity check. grid sizes and geotransforms should be the same
    assert gt_coastline == gt_rock
    assert array_coastline.shape == array_rock.shape

    # Set up output values and arrays
    array_out_shape = (int(array_rock.shape[0] / 100), int(array_rock.shape[1] / 100))
    array_ice_fraction = numpy.zeros(array_out_shape, dtype=numpy.float32)
    array_rock_fraction = numpy.zeros(array_out_shape, dtype=numpy.float32)
    array_ocean_fraction =  numpy.zeros(array_out_shape, dtype=numpy.float32)
    array_ice_mask = numpy.zeros(array_out_shape, dtype=numpy.uint8)

    # Loop through all our pixels, compute values.
    for i in range(array_out_shape[0]):
        for j in range(array_out_shape[1]):
            N = 100*100 # Number of total pixels in this sub-array
            subarray_coast = array_coastline[i*100:((i+1)*100), j*100:((j+1)*100)]
            subarray_rock  = array_rock     [i*100:((i+1)*100), j*100:((j+1)*100)]

            # There are *some* places (outlying islands) that have rock pixels that
            # are not within the coastline polygons. Flag those as inside the coastline polygons
            subarray_coast = numpy.logical_or((subarray_coast != ndv_coastline), (subarray_rock != ndv_rock))

            N_coast = numpy.count_nonzero(subarray_coast != 0)
            N_rock  = numpy.count_nonzero(subarray_rock  != ndv_rock)

            fraction_rock = float(N_rock) / float(N)
            fraction_ice  = float(N_coast - N_rock) / float(N)
            fraction_ocean = float(N - N_coast) / float(N)

            # print (i,j, fraction_ice, fraction_rock, fraction_ocean)

            if fraction_ice < 0:
                print(i,j, N_coast, N_rock)

            # Sanity checks on values:
            assert 0.0 <= fraction_rock  <= 1.0
            assert 0.0 <= fraction_ice   <= 1.0
            assert 0.0 <= fraction_ocean <= 1.0
            # Should add up to 100% with a bit of rounding tolerance
            assert 0.999 <= (fraction_rock + fraction_ice + fraction_ocean) <= 1.001

            # assign to output arrays
            array_ice_fraction[i,j] = fraction_ice
            array_rock_fraction[i,j] = fraction_rock
            array_ocean_fraction[i,j] = fraction_ocean

            array_ice_mask[i,j] = 0 if (fraction_ice < 0.50) else 1

    # Now, save our output files.

    driver = gdal.GetDriverByName("GTiff")
    SPATIAL_REFERENCE_S = osr.SpatialReference()
    SPATIAL_REFERENCE_S.ImportFromEPSG(3412) # NSIDC Sea Ice South polar stereo

    # Same spatial dimensions for geo-transform, just pixel size 100x bigger
    gt_output = (gt_coastline[0], gt_coastline[1]*100, gt_coastline[2]*100,
                 gt_coastline[3], gt_coastline[4]*100, gt_coastline[5]*100)

    for outname, outarray in zip(("ice_fraction", "rock_fraction", "ocean_fraction", "ice_mask"),
                                 (array_ice_fraction, array_rock_fraction, array_ocean_fraction, array_ice_mask)):
        out_gtif = os.path.join(output_dir, outname + ".tif")
        print ("Writing", out_gtif)
        ds = driver.Create(out_gtif, outarray.shape[1], outarray.shape[0], 1, gdal.GDT_Byte if outname == "ice_mask" else gdal.GDT_Float32)
        ds.SetGeoTransform(gt_output)
        ds.SetProjection(SPATIAL_REFERENCE_S.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(outarray)
        band.SetNoDataValue(ndv_coastline)
        outarray_float = numpy.array(outarray, dtype=numpy.float64)
        band.SetStatistics(numpy.min(outarray_float),
                           numpy.max(outarray_float),
                           numpy.mean(outarray_float),
                           numpy.std(outarray_float))
        ds.FlushCache()
        ds = None

        out_bin = os.path.splitext(out_gtif)[0] + ".bin"
        write_array_to_binary(outarray,
                              out_bin,
                              numbytes=2,
                              multiplier= (1 if (outname == "ice_mask") else 1000),
                              verbose=True)


    return

if __name__ == "__main__":
    compute_ice_masks()