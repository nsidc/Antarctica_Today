# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:03:50 2020

@author: mmacferrin
"""
import numpy

# 332 rows x 316 cols for Antarctic Polar Stereo data,
# per https://nsidc.org/data/polar-stereo/ps_grids.html
#
# If you are using Arctic data or some other grid, change the DEFAULT_GRID_SHAPE below,
# or just use the optional parameter when you call it.
# For Antarctica, (rows, cols) = (332,316)
# For Arctic, (rows, cols) = (448, 304)
DEFAULT_GRID_SHAPE = (332, 316)

def read_NSIDC_bin_file(fname,
                        grid_shape = DEFAULT_GRID_SHAPE,
                        header_size=0,
                        element_size=2,
                        return_type=float,
                        signed=False,
                        multiplier=0.1):
    """Read an SSMI file, return a 2D grid of integer values.

    header_size - size, in bytes, of the header. Defaults to zero for
        brightness-temperature data, but can be one for other data. For instance,
        NSIDC sea-ice concentration data has a 300-byte header on it.

    element_size - number of bytes for each numerical element. The brightness-temperature
        uses 2-byte little-endian integers (with a multiplier factor to turn them into floating-point values).
        NSIDC sea-ice concentration data is just 1-byte integers.

    return_type can be "int" or "float", or the numpy equivalent therein.

    signed - Whether the data values are signed (True) or unsigned (False) data

    multiplier -- A value to multiply the dataset by after it's read. Ingored for integer arrays,
        useful in some cases for floating point values that are saved as integers but then
        multiplied by 0.1 to get floating-point values.
        (Example: value "2731" with a multiplier of 0.1 will return 273.1)
        This is ingored if the return type is "int" or a numpy integer type.
    """
    # Data is in two-byte, little-endian integers, array of size GRID_SHAPE
    with open(fname, 'rb') as fin:
        raw_data = fin.read()
        fin.close()

    # Lop off the header from the start of the byte array.
    if header_size > 0:
        raw_data = raw_data[header_size:]

    # Check to make sure the data is the right size, raise ValueError if not.
    # TODO: The NSIDC-0051 data has the rows,cols in the header. We could read it from there,
    # although right now we just get the grid size from the paramter.
    if int(len(raw_data) / element_size) != int(numpy.product(grid_shape)):
        raise ValueError("File {0} has {1} elements, does not match grid size {2}.".format(
                         fname, int(len(raw_data)/element_size), str(grid_shape)))

    # Creating a uint16 array to read the data in
    int_array = numpy.empty(grid_shape, dtype=return_type)
    int_array = int_array.flatten()

    # Read the data. The built_int "from_bytes" function does the work here.
    for i in range(0, int(len(raw_data)/element_size)):
        int_array[i] = int.from_bytes(raw_data[(i*element_size):((i+1)*element_size)],
                                      byteorder="little",
                                      signed=signed)

    # Unflatten the array back into the grid shape
    int_array.shape = grid_shape

    # If the file is meant to be an integer array, just return it.
    if return_type in (int, numpy.int, numpy.uint8, numpy.int8, numpy.uint16, numpy.int16, numpy.uint32, numpy.int32, numpy.uint64, numpy.int64):
        return_array = numpy.array(int_array, dtype=return_type)
    # Else, if it's meant to be a floating-point array, scale by the multiplier
    # and return the floating-point array. If the mutiplier is a float (i.e. 0.1),
    # numpy will conver and return an array of floats
    else:
        return_array = numpy.array( int_array * multiplier, dtype=return_type)

    return return_array

if __name__ == "__main__":
    # Testing this out on a few files, examples using different types of data products.

    # An NSIDC-0001 brightness-temperature file, in 2-byte little-endian integers
    # converted to floating point. No header.
    array1 = read_NSIDC_bin_file("../Tb/nsidc-0001/tb_f08_19870709_v5_s19h.bin",
                                 grid_shape=(332,316),
                                 header_size=0,
                                 element_size=2,
                                 return_type=float,
                                 signed=False,
                                 multiplier=0.1)

    print(array1.shape, array1.dtype)
    print(array1)

    # An NSIDC-0051 sea-ice concentration v1 file, in a 1-byte unsigned integer array with
    # a 300-byte header.

    # For an Arctic file
    array2 = read_NSIDC_bin_file("../Tb/nsidc-0051/nt_20201231_f17_v1.1_n.bin",
                                 grid_shape=(448, 304),
                                 header_size=300,
                                 element_size=1,
                                 return_type=int,
                                 signed=False)

    print(array2.shape, array2.dtype)
    print(array2)

    # For an Antarctic file
    array3 = read_NSIDC_bin_file("../Tb/nsidc-0051/nt_20201231_f17_v1.1_s.bin",
                                 grid_shape=(332,316),
                                 header_size=300,
                                 element_size=1,
                                 return_type=int,
                                 signed=False)

    print(array3.shape, array3.dtype)
    print(array3)

    # An NSIDC-0079 sea-ice concentration v3 files, in 2-byte unsigned integer array with
    # a 300-byte header.

    # For an Arctic file, returning the array in integer values.
    array4 = read_NSIDC_bin_file("../Tb/nsidc-0079/bt_20201231_f17_v3.1_n.bin",
                                 grid_shape=(448, 304),
                                 header_size=0,
                                 element_size=2,
                                 return_type=int,
                                 signed=False)

    print(array4.shape, array4.dtype)
    print(array4)

    # For an Antarctic file, alternately returning the array in floating-point values (your choice, just pick the parameter you want.)
    array5 = read_NSIDC_bin_file("../Tb/nsidc-0079/bt_20201231_f17_v3.1_s.bin",
                                 grid_shape=(332,316),
                                 header_size=0,
                                 element_size=2,
                                 return_type=float,
                                 signed=False,
                                 multiplier=0.1)

    print(array5.shape, array5.dtype)
    print(array5)
