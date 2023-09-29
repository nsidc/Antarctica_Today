"""
Created on Tue May  5 12:25:44 2020

extrapolate_thermap_readings_off_map_edge.py
Read the thermap 10 m regridded (25 km) temperatures, where a few pixels are missed
off the end of the grid. Regrid it with all the same data, except using extrapolated
values for the 10 m temps off the grid.

Also, reset the NoDataValue from whatever odd floating-point value it is, to 0.0.

@author: mmacferrin
"""

import os

import numpy
from osgeo import gdal

ice_mask_tif = "F:/Research/DATA/Antarctica_Today/baseline_datasets/ice_mask.tif"
thermap_tif = "C:/Users/mmacferrin/Dropbox/Research/Antarctica_Today/Dan Dixon/derived/polar_grid_10m_temps_K_25km_EPSG3412.tif"
thermap_tif_out = os.path.join(
    os.path.split(thermap_tif)[0], "polar_grid_10m_temps_K_filled_25km_EPSG3412.tif"
)

im_ds = gdal.Open(ice_mask_tif, gdal.GA_ReadOnly)
im_array = im_ds.GetRasterBand(1).ReadAsArray()

tm_ds = gdal.Open(thermap_tif, gdal.GA_ReadOnly)
tm_gt = tm_ds.GetGeoTransform()
tm_prj = tm_ds.GetProjection()
tm_band = tm_ds.GetRasterBand(1)
tm_ndv = tm_band.GetNoDataValue()
tm_datatype = tm_band.DataType
tm_array = tm_band.ReadAsArray()

# Substitute the old NDV for the new NDV (0.0)
out_array = tm_array.copy()
out_ndv = 0.0
out_array[out_array == tm_ndv] = out_ndv

print(numpy.where(numpy.logical_and((im_array == 1), (tm_array == tm_ndv))))

# These are the hand-selected pixel values, eight lines total.
# Five going vertically along Queen Maud Land, extrapolating 1-2 pixels
# Three going horizontally along far East Antarctica, extrapolating 2-3 pixels.
# 15 pixels total extrapolated.
# The first set in each pair of lists is the indices of the points to use to
# create the quadratic model along that line. The second set is the indices of
# the 1-3 pixels over which to extrapolate.
lines_to_extrapolate_i = [
    [list(range((86 + 8), 86, -1)), [86, 85]],
    [list(range((86 + 8), 86, -1)), [86, 85]],
    [list(range((86 + 8), 86, -1)), [86]],
    [list(range((86 + 8), 86, -1)), [86]],
    [list(range((86 + 8), 86, -1)), [86]],
    [[184] * 8, [184] * 2],
    [[185] * 8, [185] * 3],
    [[186] * 8, [186] * 3],
]
lines_to_extrapolate_j = [
    [[156] * 8, [156, 156]],
    [[157] * 8, [157, 157]],
    [[158] * 8, [158]],
    [[159] * 8, [159]],
    [[160] * 8, [160]],
    [list(range((264 - 8), 264, 1)), [264, 265]],
    [list(range((264 - 8), 264, 1)), [264, 265, 266]],
    [list(range((264 - 8), 264, 1)), [264, 265, 266]],
]


# Quadratic function
def f(x, a, b, c):
    return a * (x**2) + b * x + c


for (known_i, interp_i), (known_j, interp_j) in zip(
    lines_to_extrapolate_i, lines_to_extrapolate_j
):
    # Have the x-values just go from 0 to the length of the array. Then beyond from there.
    known_x = list(range(len(known_i)))
    interp_x = [i + known_x[-1] + 1 for i in list(range(len(interp_i)))]

    # Fit a 2nd-order polynomial line (quadratic fit) to the values.
    p = numpy.polyfit(known_x, tm_array[known_i, known_j], 2)
    extrapolated_values = f(numpy.array(interp_x), *p)

    print("\n", known_i, interp_i, known_j, interp_j)
    print(tm_array[known_i, known_j], extrapolated_values)
    print(known_x, interp_x)

    # Fill in missing values with extrapolated values
    out_array[interp_i, interp_j] = extrapolated_values

# Create output GeoTiff and write all this out.
driver = tm_ds.GetDriver()
ds_out = driver.Create(
    thermap_tif_out, out_array.shape[1], out_array.shape[0], 1, tm_datatype
)

ds_out.SetGeoTransform(tm_gt)

ds_out.SetProjection(tm_prj)

band_out = ds_out.GetRasterBand(1)
band_out.WriteArray(out_array)
band_out.SetNoDataValue(out_ndv)
out_array_data = numpy.array(out_array[out_array != out_ndv], dtype=numpy.float64)
band_out.SetStatistics(
    numpy.min(out_array_data),
    numpy.max(out_array_data),
    numpy.mean(out_array_data),
    numpy.std(out_array_data),
)

ds_out.FlushCache()
band_out = None
ds_out = None
print("\n", thermap_tif_out, "written.")
