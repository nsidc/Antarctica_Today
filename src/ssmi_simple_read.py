# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:03:50 2020

@author: mmacferrin
"""
import numpy

DEFAULT_GRID_SHAPE = (332, 316) # 332 rows x 316 cols, per https://nsidc.org/data/polar-stereo/ps_grids.html

def read_SSMI_data(fname, grid_shape = DEFAULT_GRID_SHAPE, return_type=float):
    '''Read an SSMI file, return a 2D grid of integer values.'''

    # Data is in two-byte, little-endian integers, array of size GRID_SHAPE
    with open(fname, 'rb') as fin:
        raw_data = fin.read()
        fin.close()

    # Check to make sure the data is the right size, raise ValueError if not.
    if int(len(raw_data) / 2) != int(numpy.product(grid_shape)):
        raise ValueError("File {0} has {1} elements, does not match grid size {2}.".format(
                         fname, int(len(raw_data)/2)), str(grid_shape))

    # Creating a uint16 array to read the data in
    int_array = numpy.empty(grid_shape, dtype=numpy.int16)
    int_array = int_array.flatten()

    # Read the data
    for i in range(0, int(len(raw_data)/2)):
        # int_array[i] = int.from_bytes(raw_data[i:i+2], byteorder='little')
        int_array[i] = int.from_bytes(raw_data[(i*2):((i*2)+2)], byteorder="little")

    # Unflatten the array back into the grid shape
    int_array.shape = grid_shape

    if return_type in (int, numpy.int, numpy.int16, numpy.int32):
        return int_array
    else:
        return int_array / 10.0

def test():
    '''Just test and make a plot to check this is working.'''
    import matplotlib.pyplot as plt

    test_fname = r"F:/Research/DATA/Antarctica Today/Tb/nsidc-0001/tb_f08_19870709_v5_s19h.bin"

    array = read_SSMI_data(test_fname)
    norm_scale = plt.Normalize()
    norm_scale.autoscale(array)

    print(numpy.min(array), numpy.max(array),
          numpy.mean(array), numpy.median(array),
          numpy.percentile(array, 90))

    plt.cla()
    plt.hist(array.flatten(),bins=20)

    a, f = plt.subplots()
    plt.imshow(array, cmap='viridis', norm=norm_scale)

if __name__ == "__main__":
    test()