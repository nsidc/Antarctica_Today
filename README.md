# Antarctica_Today
The "Antarctica Today" code and datasets necessary to create the database, update it, and generate plots and maps of results.

## Directories:

**/baseline_datasets/** Datasets necessary for data generation, including the ice mask and region-area masks.

**/data/** -- Storing all derived data products. (Source data from NSIDC is stored in the /Tb/ directory.)

**/data/annual_sum_geotiffs/** -- GeoTiff files of the annual sum of melt days over the continent.

**/data/gap_fill_data/** -- Data of computed mean climatologies used to fill missing gaps early in the datasets (especially in the 1980s), as well as occasional missing days/values in recent datasets.

**/data/mean_climatology/** -- Data for comuting the mean climatologies.

**/data/thresholds/** -- The annual Tb-threshold files (from Mote, et al.), delineating the threshold above which melt is nominally detected in the 37H microwave brightness temperatures.

**/plots/** -- Directories for placing output line plots and maps.

**/plots/annual_maps_sum/** -- After each melt season is finished, annual maps of the sum of melt days are generated and put here.

**/plots/annual_maps_anomaly/** -- Annual maps of the anomaly from baseline averages of melt days.

**/plots/annual_line_plots/** -- Annual line graphs of melt compared to the baseline data period (1990-2020).

**/plots/daily_maps/** -- A place to put the daily-generated melt maps (empty for now).

**/Tb/** -- Raw NSIDC Tb datasets (nsidc-0080, etc)

**/qgis/** -- QGIS project that includes these datasets, for viewing. Also the vector Antarctica outline (from Quantarctica) for viewing and creating export maps. 

**/qgis/basemap_picklefiles/** -- A place to cache pickled versions of the basemaps. This increases efficiency of generating new basemaps with data plotted atop them.

**/qgis/basins/** -- Shapefiles of the basin outlines (for generating maps).

**/src/** -- Source code.

**/src/conf_int/** -- utilities for generating confidence intervals and prediction intervals on long-term trend lines.

**/src/src_baseline/** -- Source code used for generating the baseline datasets needed for the Antarctica Today dataset (namely, the gridded elevation, ice mask, and average annual temperature datasets).

## Source modules
All files outlined here are contained in the **/src/** directory:
