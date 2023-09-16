# -*- coding: utf-8 -*-

# Quick utility to read (and plot) elevations of Thermap sample points from the REMA DEM.

import os

import numpy
import pandas as pd
from matplotlib import pyplot as plt

EGM96 = True

if EGM96:
    REMA_DEM = r"F:/Research/DATA/REMA/v1.1/1km/REMA_1km_dem_filled_EGM96_minus_nd.tif"
else:
    REMA_DEM = r"F:/Research/DATA/REMA/v1.1/1km/REMA_1km_dem_filled.tif"
THERMAP_CSV = r"C:/Users/mmacferrin/Dropbox/Research/Antarctica_Today/Dan Dixon/10m_temps_ALL_Dixon.csv"

thermap_df = pd.read_csv(THERMAP_CSV, header=0)
# Add a column for REMA_Elev
thermap_df["REMA_Elev"] = [0] * len(thermap_df)
thermap_df["REMA_or_Thermap_Elev"] = [0] * len(thermap_df)

print(thermap_df.columns)

elevs = thermap_df["Elev"]
lons = thermap_df["Lon(W)"]
lats = thermap_df["Lat(S)"]


def get_rema_elev(row):
    pass


rema_elevs = [numpy.nan] * len(thermap_df)
rema_or_thermap_elevs = [numpy.nan] * len(thermap_df)

for idx, row in thermap_df.iterrows():
    return_line = os.popen(
        "gdallocationinfo -wgs84 -valonly {0} {1} {2}".format(
            REMA_DEM, row["Lon(W)"], row["Lat(S)"]
        )
    )
    try:
        print(
            "{0:>30s} {1:0.2f} {2:0.2f}, {4:0.2f}*C, {3:0.1f} -> ".format(
                row["Name"], row["Lat(S)"], row["Lon(W)"], row["Elev"], row["Temp"]
            ),
            end="",
        )
        elev_value = float(return_line.read())
        row["REMA_Elev"] = float(elev_value)
        if row["REMA_Elev"] == -9999.0:
            row["REMA_Elev"] = numpy.nan
            row["REMA_or_Thermap_Elev"] = row["Elev"]

        else:
            row["REMA_or_Thermap_Elev"] = row["REMA_Elev"]
    except ValueError:
        # Returns nothing if this point lays outside the REMA grid, breaks at float conversion
        row["REMA_Elev"] = numpy.nan
        row["REMA_or_Thermap_Elev"] = row["Elev"]

    rema_elevs[idx] = row["REMA_Elev"]
    rema_or_thermap_elevs[idx] = row["REMA_or_Thermap_Elev"]
    print("{0:0.1f}".format(row["REMA_Elev"]))

thermap_df["REMA_Elev"] = rema_elevs
thermap_df["REMA_or_Thermap_Elev"] = rema_or_thermap_elevs

print("Done")

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_aspect("equal")
ax.scatter(thermap_df["Elev"], thermap_df["REMA_Elev"])
ax.plot([-10, 3600], [-10, 3600], c="red")
ax.set_xlabel("Thermap Elevation (m)")
ax.set_ylabel("REMA Elevation (m)")
plt.tight_layout()

base, ext = os.path.splitext(THERMAP_CSV)

if EGM96:
    output_csv = base + "_REMA_EGM96" + ext
    fig_outfile = os.path.join(
        os.path.split(THERMAP_CSV)[0], "Thermap_vs_REMA_elevations_EGM96.png"
    )
else:
    output_csv = base + "_REMA" + ext
    fig_outfile = os.path.join(
        os.path.split(THERMAP_CSV)[0], "Thermap_vs_REMA_elevations.png"
    )

fig.savefig(fig_outfile, dpi=120)
print(os.path.split(fig_outfile)[1], "saved.")

thermap_df.fillna("", inplace=True)
thermap_df.to_csv(output_csv, index=False, header=True)
print(os.path.split(output_csv)[1], "saved.")
