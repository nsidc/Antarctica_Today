import os

# IMPORTANT: If we don't specify this setting, then the projection we want to use will
# be replaced with another (and this warning will be printed)!
#
#   Warning 1: CRS EPSG:3411 is deprecated. Its non-deprecated replacement EPSG:3413 will be
#   used instead. To use the original CRS, set the OSR_USE_NON_DEPRECATED configuration
#   option to NO.
#   Warning 1: CRS EPSG:3412 is deprecated. Its non-deprecated replacement EPSG:3976 will be
#   used instead. To use the original CRS, set the OSR_USE_NON_DEPRECATED configuration
#   option to NO.
os.environ["OSR_USE_NON_DEPRECATED"] = "NO"
