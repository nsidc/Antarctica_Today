#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:09:37 2021

@author: Mike MacFerrin

line_smooth.py -- just a handy function for smoothing lines in data using a convolutional filter.
"""
import numpy

def line_smooth(y, window_size, method="linear"):
    """Smooth a data vector using a convolutional kernel of length "window_size".

    Suggestion: use an odd number for "window_size".

    It will work if "window_size" is even, but the window will be slightly off-centered by one.
    In this case, it will be off-centered at the start of the run.
    e.g: window_size = 4 will center a 4-element box with 2 tailing spots before the center value and 1 tailing spot after the center.
    It'll still compute correctly, it just won't be an even window. Up to you.

    Implemented "method"s are "linear" or "gaussian".
    "gaussian" makes a gaussian kernel going out 2 std-devs (95%) with the mean in the middle.

    As currently implemented, it works only on uniformly-spaced datasets with no masked (nan) values.

    The window_size should be less than or equal to the length of the data (y). Behavior has not
    been tested if it's greater.
    """
    ws = window_size

    # Define the shape of the smoothing kernel.
    if method=="linear":
        # Divide by the size so that all kernel values add up to 1.0
        box = numpy.ones(ws)/ws

    elif method=="gaussian":
        # Make mu (mean) the middle of the window, and sigma one half of one-half
        # the length (giving it a distrigbution of 2 sigmas on either side of mu).
        x = numpy.arange(ws, dtype=numpy.float)
        mu = float(int(ws/2))
        sig = int(ws/2)/2.0 # The two divisions are purposeful.
        # If for some stupid reasion they pick a window-size of one, make sure the function doesn't break.
        if sig==0:
            sig=1
        # N==4 --> sig=1; N==5 --> sig=1; N==6 --> sig=1.5; N==7 --> sig=1.5, and so on.

        # Create a gaussian kernel peaking at 1.0 over the middle of the kernel.
        box = numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sig, 2.)))
        # Normalize the gaussial kernel so all values add up to 1.0
        box = box / numpy.sum(box)

    else:
        raise NotImplementedError("Unknown transform in 'line_smooth': " + str(method))

    # Convolve the kernel over the data.
    y_smooth = numpy.convolve(y, box, mode='same')

    # The numpy.convolve method leaves trailing edges. Fix these by scaling them
    # back up by whatever fraction of the kernel you are using.
    # Look up https://en.wikipedia.org/wiki/Convolution to see how convlution works, this makese sense.
    for i in range(int(ws/2)):
        # Scale the convolved points at the trailing front end
        y_smooth[i]      = y_smooth[i]      * numpy.sum(box) / numpy.sum(box[:(int(numpy.ceil(ws/2.)+i))])
        # Scale the convolved points at the trailing back end
        y_smooth[-(i+1)] = y_smooth[-(i+1)] * numpy.sum(box) / numpy.sum(box[(-(int(numpy.floor(ws/2))+1+i)):])

    return y_smooth
