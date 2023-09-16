# -*- coding: utf-8 -*-
"""
Created on Tue May  5 09:47:38 2020

resize_grid_by_1000.py -- Just a quick temporary script for resizing grid value
of Dan Dixon's 10m temp plot to be km instead of mis-scaled as m.

@author: mmacferrin
"""

import os

from osgeo import gdal

infile = "C:/Users/mmacferrin/Dropbox/Research/Antarctica_Today/Dan Dixon/derived/polar_grid_10m_temps_25m_OFF_BY_1000.tif"
outfile = os.path.join(
    os.path.split(infile)[0], "polar_grid_10m_temps_25km_EPSG3031.tif"
)

ds_in = gdal.Open(infile, gdal.GA_ReadOnly)

gt_in = ds_in.GetGeoTransform()
band_in = ds_in.GetRasterBand(1)
array_in = band_in.ReadAsArray()
prj_in = ds_in.GetProjection()
ndv_in = band_in.GetNoDataValue()
datatype_in = band_in.DataType
stats_in = band_in.GetStatistics(True, True)

# GeoTiff driver
driver = ds_in.GetDriver()

ds_out = driver.Create(outfile, array_in.shape[1], array_in.shape[0], 1, datatype_in)

# Multiply scale by 1000, both for corners and pixel sizes
gt_out = [value * 1000.0 for value in gt_in]
ds_out.SetGeoTransform(gt_out)

ds_out.SetProjection(prj_in)

band_out = ds_out.GetRasterBand(1)
band_out.WriteArray(array_in)
band_out.SetNoDataValue(ndv_in)
band_out.SetStatistics(*stats_in)

ds_out.FlushCache()
band_out = None
ds_out = None
print(outfile, "written.")
