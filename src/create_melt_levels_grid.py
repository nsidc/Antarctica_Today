#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:15:52 2021

create_melt_levels_grid.py -- a simple script for creating a grid of melt levels from all the different years, using output maps.

@author: mmacferrin
"""
import os
import numpy
from PIL import Image, ImageDraw, ImageFont

map_dir = "/home/mmacferrin/Research/DATA/Antarctica_Today/plots/v2.5/annual_maps/"
map_ext = ".jpg"

map_file_list = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if os.path.splitext(f)[-1].lower() == map_ext]
map_file_list.sort()

def get_year_from_filename(fname):
    """Given a file name or path, get the year from the first four characters."""
    return int(os.path.split(fname)[-1][0:4])

def get_level_from_filename(fname):
    """Given a file name or path, get the melt-threshold level from it."""
    return int(os.path.splitext(fname)[0][-1])

# Get a list of the years (y-axis) and the levels (x-axis)
map_years = list(numpy.unique([get_year_from_filename(fname) for fname in map_file_list]))
map_years.sort()

map_levels = list(range(8,1,-1))

# Open one of the images and get the file dimensions.
im_size = Image.open(map_file_list[0]).size
print(im_size)
offset_size = 100
master_image_size = (im_size[0]*len(map_years)  + offset_size + offset_size,
                     im_size[1]*len(map_levels) + offset_size + offset_size,                     )

master_image = Image.new("RGB", master_image_size, color=(255,255,255))
print(master_image.size)

# Paste each of the maps into the image.
for map_fname in map_file_list:
    print(os.path.split(map_fname)[1])
    map_year = get_year_from_filename(map_fname)
    map_level = get_level_from_filename(map_fname)
    map_image = Image.open(map_fname)

    x_corner = im_size[0]*map_years.index(map_year) + offset_size
    y_corner = im_size[1]*map_levels.index(map_level) + offset_size

    master_image.paste(map_image, box=(x_corner, y_corner, x_corner+im_size[0], y_corner+im_size[1]))

# Load font from my OS, randomly choosing Ubuntu Bold here.
font_file = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
im_font = ImageFont.truetype(font_file, 60)
d = ImageDraw.Draw(master_image)

# Add vertical labels for levels
for i,map_level in enumerate(map_levels):
    # Create a temporary small image to write the original text in.
    txt_img = Image.new("RGB", (300, 70), color=(255,255,255))
    txt_d = ImageDraw.Draw(txt_img)

    txt_d.text((0,0),
               "Level {0}".format(map_level) if (map_level < 8) else "(No Filter)",
               fill=(0,0,0),
               font=im_font)

    txt_rotated_img = txt_img.transpose(Image.ROTATE_90)

    corner_x = 20
    corner_y = i*im_size[1] + offset_size + 120

    left_box = (corner_x, corner_y, corner_x + txt_rotated_img.size[0], corner_y + txt_rotated_img.size[1])
    master_image.paste(txt_rotated_img, left_box)
    right_box = (master_image.size[0] - 90, left_box[1], master_image.size[0] - 90 + txt_rotated_img.size[0], left_box[3])
    master_image.paste(txt_rotated_img, right_box)

# Add horizontal labels for years
for j,map_year in enumerate(map_years):
    year_txt = "{0}-{1}".format(map_year, map_year+1)
    # Top margin
    corner_x = j*im_size[0] + offset_size + 230
    corner_y = 20
    d.text((corner_x, corner_y),
           year_txt,
           fill=(0,0,0),
           font=im_font)

    # Bottom margin, simply cut the text from the top margin
    top_box = (corner_x, corner_y, corner_x + 500, corner_y + 70)
    bottom_box = (top_box[0], master_image.size[1] - 80, top_box[2], master_image.size[1] - 10)
    text_region = master_image.crop(top_box)
    # Past the region back into the bottom box.
    master_image.paste(text_region, bottom_box)

master_image_fname = os.path.join(os.path.split(os.path.split(map_file_list[0])[0])[0], "melt_levels_grid.jpg")
master_image.save(master_image_fname)

print(master_image_fname, "written.")