# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:13:01 2020

@author: mmacferrin
"""
import gdal
import numpy
import os
import argparse

def resize_tif_to_reference_grid(gtif_in,
                                 gtif_reference,
                                 gtif_out,
                                 verbose=False):
    '''I have RACMO & REMA files written out the same grid format & resolution
    as the NSIDC's nsidc-0001 and nsidc-0080 files. But the grid sizes are different
    with different boundaries. This takes a .tif GeoTiff, and a reference Tb GeoTiff,
    and creates a copy of the gtif_in data with the same array size as the gtif_Tb_reference,
    and spits it out to gtif_out. Extra values are filled in with the gtif_in NoDataValue.
    '''
    if verbose:
        print("Reading", os.path.split(gtif_in)[1])

    ds_in = gdal.Open(gtif_in, gdal.GA_ReadOnly)
    if ds_in is None:
        raise FileNotFoundError("Gdal could not read input file '{0}'".format(gtif_in))

    if verbose:
        print("Reading", os.path.split(gtif_reference)[1])

    ds_ref = gdal.Open(gtif_reference, gdal.GA_ReadOnly)
    if ds_ref is None:
        raise FileNotFoundError("Gdal could not read reference file '{0}'".format(gtif_reference))

    geotransform_in = ds_in.GetGeoTransform()
    band_in = ds_in.GetRasterBand(1)
    array_in = band_in.ReadAsArray()

    geotransform_ref = ds_ref.GetGeoTransform()
    band_ref = ds_ref.GetRasterBand(1)
    array_ref = band_ref.ReadAsArray()

    # Check to make sure the grids have the same x- and y- resolutions
    if not (geotransform_in[1:3] == geotransform_ref[1:3] and \
            geotransform_in[4:6] == geotransform_ref[4:6]):
        raise ValueError("Input resolutions do not match.")

    x_res_in = geotransform_in[1]
    x_res_ref = geotransform_ref[1]

    y_res_in = geotransform_in[5]
    y_res_ref = geotransform_ref[5]

    x_UL_in = geotransform_in[0]
    y_UL_in = geotransform_in[3]

    x_UL_ref = geotransform_ref[0]
    y_UL_ref = geotransform_ref[3]

    # Check to make sure the grids are aligned exactly atop one another.
    if not ((x_UL_in % x_res_in) == (x_UL_ref % x_res_ref) and \
            (y_UL_in % y_res_in) == (y_UL_ref % y_res_ref)):
        print("X: {0} % {1} = {2}, {3} % {4} = {5}".format(x_UL_in, x_res_in,
                                                           x_UL_in % x_res_in,
                                                           x_UL_ref, x_res_ref,
                                                           x_UL_ref % x_res_ref))
        print("Y: {0} % {1} = {2}, {3} % {4} = {5}".format(y_UL_in, y_res_in,
                                                           y_UL_in % y_res_in,
                                                           y_UL_ref, y_res_ref,
                                                           y_UL_ref % y_res_ref))
        raise ValueError("Input grids are not geographically aligned.")

    # Create the output array, same shape as the reference array, but same datatype
    # as the source array. Fill with the array_in NDV
    array_out = numpy.zeros(array_ref.shape, dtype=array_in.dtype) + band_in.GetNoDataValue()

    # Calculate the pixel offsets in x and y direction
    x_offset = int((x_UL_in - x_UL_ref) / x_res_in)
    y_offset = int((y_UL_in - y_UL_ref) / y_res_in)

    # Check if the input data is going to get clippsed in any of the sides.
    if x_offset < 0:
        raise UserWarning("Input data clipped on the left.")
        array_in = array_in[:, -x_offset:]
        x_offset = 0

    if y_offset < 0:
        raise UserWarning("Input data clipped on the top.")
        array_in = array_in[-y_offset:, :]
        y_offset = 0

    if (x_offset + array_in.shape[1]) > array_out.shape[1]:
        raise UserWarning("Input data clipped on the right.")
        array_in = array_in[:, x_offset:array_out.shape[1]]

    if (y_offset + array_in.shape[0]) > array_out.shape[0]:
        raise UserWarning("Input data clipped on the bottom.")
        array_in = array_in[y_offset:array_out.shape[0], :]

    # Copy the data over.
    array_out[y_offset : y_offset + array_in.shape[0],
              x_offset : x_offset + array_in.shape[1]] = array_in

    # Create the output dataset
    driver = ds_in.GetDriver()
    ds_out = driver.Create(gtif_out, array_out.shape[1], array_out.shape[0], 1, band_in.DataType)
    if ds_out is None:
        raise IOError("GDAL could not create", os.path.split(gtif_out)[-1])
    ds_out.SetGeoTransform(geotransform_ref)
    ds_out.SetProjection(ds_in.GetProjection())

    band_out = ds_out.GetRasterBand(1)
    band_out.WriteArray(array_out)
    band_out.SetNoDataValue(band_in.GetNoDataValue())

    # Calculate band statistics
    good_data_out = array_out[array_out != band_in.GetNoDataValue()]
    band_out.SetStatistics(float(numpy.min(good_data_out)),
                           float(numpy.max(good_data_out)),
                           float(numpy.mean(good_data_out)),
                           float(numpy.std(good_data_out)))
    ds_out.FlushCache()
    ds_out = None

    if verbose:
        print(os.path.split(gtif_out)[-1], "written.")

    return

def read_and_parse_args():
    parser = argparse.ArgumentParser(description="Resizes a geotif to have the same boundaries as a reference geotif, and outputs the newly-sized output geotif. CONDITION: The two geotifs should already be the same grid resolution and offset. This doesn't do any resampling, just resizes the grids.")
    parser.add_argument("input_gtif", type=str, help="Source file (.tif)")
    parser.add_argument("reference_gtif", type=str, help="Reference file (.tif)")
    parser.add_argument("output_gtif", type=str, help="Destination file (.tif)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Increase output verbosity.")

    return parser.parse_args()

if __name__ == "__main__":
    args = read_and_parse_args()

    resize_tif_to_reference_grid(args.input_gtif,
                                 args.reference_gtif,
                                 args.output_gtif,
                                 verbose=args.verbose)
