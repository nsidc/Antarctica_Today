# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:21:04 2020

@author: mmacferrin
"""

import netCDF4
from osgeo import osr, gdal
import os
import numpy

import resample_grid

gdal.UseExceptions()
osr.UseExceptions()

# TODO: The below line was as-is, which is a syntax error. What should it look like?
# netCDF4.

source_file = (
    r"F:/Research/DATA/RACMO 2.3p2/Antarctica/RACMO2.3_p2_ANT27_Tskin_avg_1979-2016.nc"
)
dest_file = os.path.splitext(source_file)[0] + "_tskin.tif"
ds = netCDF4.Dataset(source_file)
proj4_str = ds.variables["rotated_pole"].proj4_params
skin_temps = ds.variables["tskin"][:]
nodata = ds.variables["tskin"].missing_value
skin_temps = skin_temps.squeeze()  # Get rid of single dimensions

print(proj4_str)

srs = osr.SpatialReference()
srs.ImportFromProj4(proj4_str)
srs.Validate()
# print(srs.ExportToWkt()) # "WKT2:2019"))

driver = gdal.GetDriverByName("GTiff")
datatype = gdal.GDT_Float32

temps_wo_nodata = skin_temps[skin_temps != nodata]

ds = driver.Create(dest_file, skin_temps.shape[1], skin_temps.shape[0], 1, datatype)
ds.SetProjection(srs)
band = ds.GetRasterBand(1)
band.WriteArray(skin_temps)
band.SetNoDataValue(nodata)
band.SetStatistics(
    numpy.min(temps_wo_nodata),
    numpy.max(temps_wo_nodata),
    numpy.mean(temps_wo_nodata),
    numpy.std(temps_wo_nodata),
)
ds.FlushCache()
ds = None
