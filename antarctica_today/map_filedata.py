# TODO: Move constants into the `constants` subpackage
import os
from pathlib import Path

import cartopy

from antarctica_today.constants.paths import DATA_BASELINE_DATASETS_DIR, DATA_QGIS_DIR

# TODO: *_name -> *_path
boundary_shapefile_fn = "Antarctic_Coastline_low_res_polygon_simplified.shp"
boundary_shapefile_path = DATA_QGIS_DIR / boundary_shapefile_fn
boundary_shapefile_reader = cartopy.io.shapereader.Reader(boundary_shapefile_path)

mountains_shapefile_fn = "LandSat8_Rock_outcrops_simplified.shp"
mountains_shapefile_path = DATA_QGIS_DIR / mountains_shapefile_fn
mountains_shapefile_reader = cartopy.io.shapereader.Reader(mountains_shapefile_path)

map_picklefile_directory = DATA_QGIS_DIR / "basemap_picklefiles"

ice_mask_tif = DATA_BASELINE_DATASETS_DIR / "ice_mask.tif"

# Dictionary to retrieve basemaps for each region.
# Keys: (map_type, region_number):
# Values: file path for the basemap picklefile.
map_picklefile_dictionary = {}
for map_type in ("daily", "annual", "anomaly"):
    for region_num in range(8):
        map_picklefile_dictionary[(map_type, region_num)] = os.path.join(
            map_picklefile_directory,
            "basemap_region_{0}_{1}.pickle".format(region_num, map_type),
        )

from tb_file_data import outputs_annual_plots_directory

annual_maps_directory = outputs_annual_plots_directory
daily_maps_directory = os.path.join(
    os.path.split(annual_maps_directory)[0], "daily_maps"
)
anomaly_maps_directory = os.path.join(
    os.path.split(annual_maps_directory)[0], "annual_maps_anomaly"
)

# A shapefile containing a vector outline for each region, separately. The {0} region just contains them all.
region_outline_shapefiles_dict: dict[int, Path] = {
    0: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2.shp",
    1: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R1.shp",
    2: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R2.shp",
    3: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R3.shp",
    4: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R4.shp",
    5: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R5.shp",
    6: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R6.shp",
    7: DATA_QGIS_DIR / "basins" / "Antarctic_Regions_v2_R7.shp",
}
