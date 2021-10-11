"""Create a simple gridded file (every 10,50,100 pixels) to overlay in QGIS and easily visualize grid coordinates."""
# -*- coding: utf-8 -*-


from ssmi_bin_to_gtif import output_gtif, GRIDSIZE_25_S
import numpy

grid_filename = "../baseline_datasets/pixel_100_grid_file.tif"

def create_grid_file(fname = grid_filename):
    """Create the file."""
    array = numpy.zeros(GRIDSIZE_25_S, dtype=numpy.int16) # (332, 316)
    range_10_rows = numpy.arange(0,array.shape[0],10)
    range_50_rows = numpy.arange(0,array.shape[0],50)
    range_100_rows = numpy.arange(0,array.shape[0],100)

    range_10_cols = numpy.arange(0,array.shape[1],10)
    range_50_cols = numpy.arange(0,array.shape[1],50)
    range_100_cols = numpy.arange(0,array.shape[1],100)

    array[range_10_rows,:] = 10
    array[:,range_10_cols] = 10
    array[range_50_rows,:] = 50
    array[:,range_50_cols] = 50
    array[range_100_rows,:] = 100
    array[:,range_100_cols] = 100

    output_gtif(array, grid_filename, nodata=0)


if __name__ == "__main__":
    create_grid_file()