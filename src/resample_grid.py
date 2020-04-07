# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:42:09 2020

@author: mmacferrin
"""

import argparse
import gdal

def regrid(source, destination, reference, gdal_resample_alg='average', verbose=True):
    '''Take the "source" file, regrid to the same grid as "reference", save to "destination."
    '''
    refDS = gdal.Open(reference, gdal.GA_ReadOnly)
    sourceDS = gdal.Open(source, gdal.GA_ReadOnly)
    geotransform = refDS.GetGeoTransform()
    # Output bounds are in (ulx, uly, lrx, lry)
    # output_bounds = [geotransform[0],
    #                  geotransform[3],
    #                  geotransform[0]+(geotransform[1]*refDS.RasterXSize),
    #                  geotransform[3]+(geotransform[5]*refDS.RasterYSize)]


    gdal.UseExceptions()
    options = gdal.TranslateOptions(maskBand="auto",
                                    bandList=[i+1 for i in range(sourceDS.RasterCount)],
                                    # width=refDS.RasterXSize,
                                    # height=refDS.RasterYSize,
                                    xRes = geotransform[1],
                                    yRes = geotransform[5],
                                    # outputBounds = output_bounds,
                                    outputSRS = refDS.GetSpatialRef(),
                                    resampleAlg = gdal_resample_alg,
                                    stats=True)
    gdal.Translate(destination, sourceDS, options=options)

    if verbose:
        print(destination, "written.")

def read_and_parse_arguments():
    parser = argparse.ArgumentParser(description="Re-grid [src] dataset to the same grid & resolution as [ref]. Save to [dest].")
    parser.add_argument('src', help="File to be regridded.")
    parser.add_argument('ref', help="Reference file with desired grid.")
    parser.add_argument('dest', help="Location of regridded destination file to write.")
    parser.add_argument('--gdal_resample_alg', type=str, default="average",
                        help="GDAL Resampling Algorithm. Defaults to 'average'.")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Increase verbosity.")

    return parser.parse_args()

if __name__ == "__main__":
    args = read_and_parse_arguments()

    regrid(args.src,
           args.dest,
           args.ref,
           args.gdal_resample_alg,
           verbose=args.verbose)