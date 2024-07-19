from typing import Final

# If you are using Arctic data or some other grid, change the DEFAULT_GRID_SHAPE below,
# or just use the optional parameter when you call it.
#
# * For Antarctica, (rows, cols) = (332,316)
# * For Arctic, (rows, cols) = (448, 304)
#
# Source: https://nsidc.org/data/polar-stereo/ps_grids.html
DEFAULT_GRID_SHAPE: Final = (332, 316)
