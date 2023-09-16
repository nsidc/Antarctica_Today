# -*- coding: utf-8 -*-

import numpy
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

thermap_csv_file = r"C:/Users/mmacferrin/Dropbox/Research/Antarctica_Today/Dan Dixon/10m_temps_ALL_Dixon_REMA_EGM96.csv"

thermap_df = pd.read_csv(thermap_csv_file, header=0)

temps = thermap_df["Temp"]
elevs = thermap_df["REMA_or_Thermap_Elev"]
lats = thermap_df["Lat(S)"]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharey=True)

axes[0].scatter(elevs, temps, color="blue")
axes[0].set_title("Temp vs Elevation")
axes[0].set_xlabel("Elevation (m)")
axes[0].set_ylabel("10 m temperature (C)")

X = sm.add_constant(elevs)
model_elev_only = sm.OLS(thermap_df[["Temp"]], X).fit()
print(model_elev_only.summary())
coefs = model_elev_only.params


def plus_minus_op(x):
    """Return '-' if the number is negative, else '+'."""
    return "-" if x < 0 else "+"


min_max_elev = numpy.array([min(elevs), max(elevs)])
axes[0].plot(
    min_max_elev,
    coefs["REMA_or_Thermap_Elev"] * min_max_elev + coefs["const"],
    color="black",
)
axes[0].text(
    0.98,
    0.98,
    "temp_C = {0:0.6f}$\cdot$elev\n{1:s} {2:0.4f}".format(
        coefs["REMA_or_Thermap_Elev"],
        plus_minus_op(coefs["const"]),
        numpy.abs(coefs["const"]),
    ),
    transform=axes[0].transAxes,
    horizontalalignment="right",
    verticalalignment="top",
    fontsize="large",
)
axes[0].text(
    0.05,
    0.06,
    "R$^2$ = {0:0.2f}".format(model_elev_only.rsquared),
    transform=axes[0].transAxes,
    horizontalalignment="left",
    fontsize="large",
)


axes[1].scatter(lats, temps, color="green")
axes[1].set_title("Temp vs Latitude")
axes[1].set_xlabel("Latitude (deg)")

X = thermap_df[["REMA_or_Thermap_Elev", "Lat(S)"]]
Y = thermap_df[["Temp"]]

print("\n=== Statsmodels ===")
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

print(model.summary())
coefs = model.params

temps_lat_corrected_75 = temps - coefs["Lat(S)"] * (75 + lats)
axes[2].scatter(elevs, temps_lat_corrected_75, color="purple")

# # Compute a quadratic curve through this line.
# poly_coefs = numpy.polyfit(elevs, temps_lat_corrected_75, deg=2)
# print(poly_coefs)
# # Quadratic trend-line
# trend_x = numpy.linspace(*min_max_elev, 100)
# trend_y = poly_coefs[0]*(trend_x**2) + poly_coefs[1]*trend_x + poly_coefs[2]
# axes[2].plot(trend_x, trend_y, color="black")

# Linear trend-line
axes[2].plot(
    min_max_elev,
    (coefs["REMA_or_Thermap_Elev"] * numpy.array(min_max_elev))
    + (-75 * coefs["Lat(S)"])
    + (coefs["const"]),
    color="black",
)

axes[2].set_title("Temp vs. Elev (corrected to 75 S lat)")
axes[2].set_xlabel("Elevation (m)")

axes[2].text(
    0.98,
    0.98,
    "temp_C = {0:0.6f}$\cdot$elev\n{3:s} {1:0.4f}$\cdot$latitude\n{4:s} {2:0.4f}".format(
        coefs["REMA_or_Thermap_Elev"],
        numpy.abs(coefs["Lat(S)"]),
        numpy.abs(coefs["const"]),
        plus_minus_op(coefs["Lat(S)"]),
        plus_minus_op(coefs["const"]),
    ),
    transform=axes[2].transAxes,
    horizontalalignment="right",
    verticalalignment="top",
    fontsize="large",
)
axes[2].text(
    0.05,
    0.06,
    "R$^2$ = {0:0.2f}".format(model.rsquared),
    transform=axes[2].transAxes,
    horizontalalignment="left",
    fontsize="large",
)

plt.tight_layout()
