# -*- coding: utf-8 -*-
"""
long_term_trends.py -- a script for analyzing long-term melt data from Antarctica Today data.
"""
import os
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib
# from matplotlib import rc
import numpy
import statsmodels
import statsmodels.api
from statsmodels.stats.outliers_influence import summary_table

matplotlib.style.use("default")

from melt_array_picklefile import get_ice_mask_array
from tb_file_data import outputs_v2_5_annual_tifs_directory, \
                         antarctic_regions_tif,              \
                         antarctic_regions_dict,             \
                         model_results_v2_5_plot_directory

def read_annual_sum_tif(year, gap_filled=True):
    """Read the total melt days from the annual sums tif.

    These annual sums are created by "compute_mean_climatology.create_annual_melt_sum_tif(year="all").

    Look in there for the file name YYYY-ZZZZ.tif .
    If 'gap_filled', look for the 'YYYY-ZZZZ_gap_filled.tif' file.
    """
    tif_name = "{0}-{1}{2}.tif".format(year, year+1, "_gap_filled" if gap_filled else "")
    tif_name = os.path.join(outputs_v2_5_annual_tifs_directory, tif_name)

    ds = gdal.Open(tif_name,gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError("Could not open {0}".format(tif_name))
    array = ds.GetRasterBand(1).ReadAsArray()
    return array

def compute_annual_extent_array(year,
                                gap_filled=False,
                                days_threshold=1):
    """Compute the annual melt extent, any pixel where melt_days >= days_threshold.

    Return as an MxN, (-1,0,1)-value grid for (no-data, melt>=threshold, melt<threshold).
    """
    melt_array = read_annual_sum_tif(year, gap_filled=gap_filled)
    mask = get_ice_mask_array()

    melt_array[melt_array < days_threshold] = 0
    melt_array[melt_array >= days_threshold] = 1
    melt_array[mask==0] = -1

    return melt_array

def get_time_series_data(region_number=0,
                         melt_index_or_extent="index",
                         start_year=1979,
                         end_year=2019,
                         extent_melt_days_threshold=1,
                         omit_1987=True,
                         gap_filled=True,
                         return_in_km2=True):
    """Return two vectors of (years, melt-each-year) for the whole time series.

    Includes years from start_year through end_year, inclusive. (Note: End year
    of 2019 means the 2019-20 melt season.)

    The year 1987-88 is missing more than 40 days of data due to faulty satellite
    instruments. Generally for a time-series, we want to omit that year (unless
    you use gap-filled data and want to include it.)

    If return_in_km2==False, return the number of pixels.
    Else if return_in_km2==True, return the km2 (for extent), or km2*days (for index).
    """
    if omit_1987 and start_year <= 1987 and end_year >= 1987:
        N = end_year - start_year
    else:
        N = end_year - start_year + 1

    years = numpy.empty((N,),dtype=numpy.int)
    melt_vector = numpy.empty((N,),dtype=numpy.int)

    if region_number == 0:
        mask = get_ice_mask_array()
    else:
        ds = gdal.Open(antarctic_regions_tif,gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError("Could not open {0}".format(antarctic_regions_tif))
        regions_array = ds.GetRasterBand(1).ReadAsArray()
        mask = (regions_array == region_number)

    melt_index_or_extent_lower = melt_index_or_extent.strip().lower()

    i=0
    for year in range(start_year,end_year+1):
        if (year == 1987) and omit_1987:
            continue

        years[i] = year

        if melt_index_or_extent_lower == "index":
            melt_array = read_annual_sum_tif(year, gap_filled=gap_filled)
        elif melt_index_or_extent_lower == "extent":
            melt_array = compute_annual_extent_array(year, gap_filled=gap_filled, days_threshold=extent_melt_days_threshold)
        else:
            raise ValueError("Unknown value for parameter 'melt_index_or_extent': {0}".format(melt_index_or_extent))

        melt_vector[i] = numpy.sum(melt_array[mask==1])

        i=i+1

    # Make sure we didn't leave any blank values.
    assert numpy.all([val != None for val in melt_vector])

    if return_in_km2:
        melt_vector = melt_vector * (25**2)

    return years, melt_vector

def plot_time_series(fname_template=None,
                     region_number="all",
                     dpi=150,
                     ax = None,
                     melt_index_or_extent="index",
                     extent_melt_days_threshold=2,
                     include_ylabel=True,
                     gap_filled=True,
                     include_trendline=False,
                     include_trendline_only_if_significant=True,
                     include_legend_if_significant=True,
                     include_name_in_title=True,
                     print_trendline_summary=True,
                     offset_years_by_one=True,
                     add_confidence_intervals=True,
                     add_prediction_intervals=True,
                     verbose=True):
    """Cretae a plot of the time series of melt.

    In fname_template, if you specify a {0} tag in the name, it will be filled
    in with the region number. This is useful if you want to use region_number="all",
    as that will create 8 plots for regions 0-7.

    Use 'ax' to provide an axis upon which to draw. This is useful for putting
    together a multi-part figure. Don't use this option if using "region_number="all",
    as that will draw multiple plots on the same axes.
    """
    if region_number == "all":
        region_nums = range(8)
    else:
        region_nums = [region_number]

    ax_provided = ax

    for region_n in region_nums:
        years, melt = get_time_series_data(region_number=region_n,
                                           melt_index_or_extent=melt_index_or_extent,
                                           extent_melt_days_threshold = extent_melt_days_threshold,
                                           gap_filled=gap_filled,
                                           return_in_km2=True)

        # Since the "2019" melt season (.e.g) in Antarctica actually spans 2019-2020,
        # it makes more sense to center it over the Jan 1, 2020 date rather than
        # the start of 2019.
        # Make it so.
        if offset_years_by_one:
            years = years + 1

        if include_ylabel:
            if max(melt) > 1e6:
                melt = melt / 1e6
                figure_exp = 6
            else:
                melt = melt / 1e3
                figure_exp = 3
        else:
            melt = melt / 1e6
            figure_exp = 6

        # Create a new figure if no axis is provided.
        if ax_provided is None:
            fig, ax = plt.subplots(1,1)

        ax.plot(years, melt, color="maroon", label = "Annual melt {0}".format("index" if melt_index_or_extent == "index" else "extent"))

        melt_index_or_extent_lower = melt_index_or_extent.strip().lower()

        if include_ylabel:
            if melt_index_or_extent_lower == "index":
                ax.set_ylabel("Melt Index (10$^{0}$ km$^2\cdot$days)".format(figure_exp))
                # ax.set_ylabel("Melt Index (million km$^2$ days)")
            elif melt_index_or_extent_lower == "extent":
                ax.set_ylabel("Melt Extent (10$^{0}$ km$^2$)".format(figure_exp))
            else:
                raise ValueError("Unknown value for parameter 'melt_index_or_extent': {0}".format(melt_index_or_extent))

        ax.tick_params(direction="in", bottom=True, left=True, right=True, top=False, labeltop=False, labelright=False, which="major")
        ax.tick_params(direction="in", bottom=True, which="minor")
        ax.tick_params(axis='x', length=4, which="major")
        ax.tick_params(axis='x', length=2, which="minor")

        if include_name_in_title:
            region_name = antarctic_regions_dict[region_n]

            ax.set_title(region_name)


        # Limit lower-bounds to zero
        ylim = ax.get_ylim()
        ax.set_ylim(max(ylim[0], 0), ylim[1])

        # Force the y-axis to only use integers (this tends to give us better scaling)
        if ylim[1] > 8:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        # Turn on the minor ticks for the years.
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=1))
        # ax.xaxis.grid(True, which='minor')

        # Run a linear-fit OLS model on the data.
        x = statsmodels.api.add_constant(years)
        model = statsmodels.api.OLS(melt, x)
        results = model.fit()

        # If go into all this if we've indicated we might want to plot a trendline.
        if include_trendline or include_trendline_only_if_significant:

            # print(results.params)
            # print(results.pvalues)
            pval_int, pval_slope = results.pvalues
            intercept, slope = results.params
            # fit_func = numpy.poly1d((slope, intercept))

            if print_trendline_summary:
               print("\n")
               print("============", antarctic_regions_dict[region_n] + ",", melt_index_or_extent, "==============")
               print(results.summary())

            if include_trendline or (pval_slope <= 0.05 and include_trendline_only_if_significant):

                st, data, ss2 = summary_table(results, alpha=0.05)
                fittedvalues = data[:, 2]
                # predict_mean_se  = data[:, 3]
                predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
                predict_ci_low, predict_ci_upp = data[:, 6:8].T

                # Put the p-value in the legend text.
                p_value_text = ("{0:0.3f}" if (pval_slope > 0.001) else "{0:0.1e}").format(pval_slope)
                # ax.plot(years, fit_func(years), color="blue", label = r"Linear Trend (\textit{p=" + p_value_text + "})")
                # ax.plot(years, fit_func(years), color="blue", label = r"Linear Trend ($\it{p=" + p_value_text + "}$)")
                ax.plot(years, fittedvalues, color="blue", label = r"Linear trend ($\it{p=" + p_value_text + "}$)")

                if add_confidence_intervals:
                    # Regression errors, Y minus Y_fit
                    # y_err = melt - fit_func(years)

                                    # Calculate confidence intervals
                    # p_x, confs = CI.conf_calc(years, y_err, c_limit=0.975, test_n=50)

                    # Calculate the lines for plotting:
                    # The fit line, and lower and upper confidence bounds
                    # p_y, lower, upper = CI.ylines_calc(p_x, confs, fit_func)

                    # plot confidence limits
                    # ax.plot(p_x, lower, 'c--',
                    ax.plot(years, predict_mean_ci_low, color='blue', linestyle='--',
                            label='95% confidence interval',
                            # label='95\% Confidence Interval',
                            alpha=0.5,
                            linewidth=0.8)
                    # ax.plot(p_x, upper, 'c--',
                    ax.plot(years, predict_mean_ci_upp, color='blue', linestyle='--',
                            label=None,
                            alpha=0.5,
                            linewidth=0.8)

                if add_prediction_intervals:
                    ax.plot(years, predict_ci_low, color="red", linestyle='--',
                            label='95% prediction interval',
                            # label='95\% Confidence Interval',
                            alpha=0.5,
                            linewidth=0.5)
                    # ax.plot(p_x, upper, 'c--',
                    ax.plot(years, predict_ci_upp, color="red", linestyle='--',
                            label=None,
                            alpha=0.5,
                            linewidth=0.5)

                    # The prediction intervals are quite wide. Rescale the y-limits
                    # to be no more than 10% above/below the max/min of the data,
                    # even if it makes the prediction intervals trail off the figure
                    # a bit.
                    ylim = ax.get_ylim()
                    if (ylim[0] < 0) or (ylim[0] < (min(melt) - 0.1*(max(melt) - min(melt)))):
                        ax.set_ylim(max(0, min(melt)- 0.1*(max(melt) - min(melt))), ylim[1])

                    ylim = ax.get_ylim()
                    if (ylim[1] > (max(melt) + 0.1*(max(melt) - min(melt)))):
                        ax.set_ylim(ylim[0], (max(melt) + 0.1*(max(melt) - min(melt))))

                if include_legend_if_significant:
                    ax.legend(fontsize="small", labelspacing=0.1, framealpha=0.95)


        if ax_provided is None:
            fig.tight_layout()

            if fname_template is None:
                plt.show()
            else:
                fname = fname_template.format(region_n)
                fig.savefig(fname, dpi=dpi)
                if verbose:
                    print(fname, "written.")

            plt.close(fig)

    return results

def special_plot_antarctica_and_peninsula_index(figname = os.path.join(model_results_v2_5_plot_directory,
                                                                       "trends","R0_R1_index_trends.png")):
    """Make a special plot for the BAMS report, just having Antarctica & the Peninsula in it."""
    fig, axes = plt.subplots(2,1, figsize=(6.4, 5.5))
    ax1, ax2 = axes

    # Plot Antarctica on top
    results = plot_time_series(region_number=0,
                     melt_index_or_extent="index",
                     ax=ax1,
                     include_name_in_title=False,
                     offset_years_by_one=False,
                     add_confidence_intervals=True,
                     add_prediction_intervals=False)
    ax1.text(0.03, 0.85, antarctic_regions_dict[0], ha="left", va="top", fontsize="x-large", transform=ax1.transAxes)

    # Plot Antarctic Peninsula on bottom
    plot_time_series(region_number=1,
                     melt_index_or_extent="index",
                     ax=ax2,
                     include_name_in_title=False,
                     offset_years_by_one=False,
                     add_confidence_intervals=True,
                     add_prediction_intervals=False)
    ax2.text(0.03, 0.85, antarctic_regions_dict[1].replace(" ","\n"), ha="left", va="top", fontsize="x-large", transform=ax2.transAxes)

    fig.tight_layout()

    for fmt in (".png", ".svg"):
        figname = os.path.splitext(figname)[0] + fmt
        fig.savefig(figname, dpi=600)
        print(figname, "written.")

    return results

def special_plot_antarctica_and_all_regions(figname = os.path.join(model_results_v2_5_plot_directory,
                                                                   "trends","ALL_regions_index_trends.png")):
    """Make a special plot for the BAMS report, having all the regions in it."""
    fig, axes = plt.subplots(2,4, sharex=True, sharey=False, figsize=(12., 4.))
    print(axes)

    for region in range(8):

        ax = axes[int(int(region)/4), int(region%4)]

        # Plot Antarctica on top
        results = plot_time_series(region_number=region,
                                   melt_index_or_extent="index",
                                   ax=ax,
                                   include_name_in_title=False,
                                   offset_years_by_one=False,
                                   add_confidence_intervals=True,
                                   add_prediction_intervals=False,
                                   include_ylabel = False,
                                   include_legend_if_significant=False)

        region_name = antarctic_regions_dict[region]
        # Adjust the region placements
        ha = "right"
        x = 0.97
        if region == 5:
            x = 0.30
            ha = "left"

        elif region==6:
            x = 0.35
            ha = "left"

        elif region == 7:
            region_name = region_name.replace(" ", "\n")

        ax.text(x, 0.97, region_name,
                ha=ha, va="top",
                fontsize="medium",
                transform=ax.transAxes)

        # Add equation if significant.
        if results.pvalues[1] <= 0.05:
            intercept, slope = results.params
            # equation_text = r"$\it{" + r"{0:0.02f}".format(slope) + r"\cdot" + " year\n+" + r"{0:0.02f}".format(intercept) + r"}$"
            equation_text = r"$\bf{y}$" + "={0:0.3g}".format(slope) + r"$\bf{x}$" + "+{0:0.3g}\n".format(intercept) + \
                            r"$\it{p}$=" + "{0:0.2g}".format(results.pvalues[1])
            ax.text(0.96, 0.80, equation_text, ha="right", va="top", transform=ax.transAxes, fontsize="small")

        # Add the plot letter.0
        letter = ['a','b','c','d','e','f','g','h'][region]
        ax.text(0.025, 0.98, letter, ha="left", va="top", transform=ax.transAxes, fontsize="medium", fontweight="bold")

    fig.text(0.001, 0.5, "Melt Index (10$^6$ km$^2\cdot$days)", rotation="vertical", size="large", ha="left", va="center", transform = fig.transFigure)

    fig.tight_layout()

    fig.subplots_adjust(left=0.04, top=0.925, hspace=0.075, wspace=0.13)

    ax = axes[0,0]
    ax.legend(bbox_to_anchor=(0.034, 1),
              loc="upper left",
              ncol=3,
              bbox_transform=fig.transFigure,
              labels=["Annual melt index","Linear trend (if significant)","95% confidence interval"],
              frameon=True,
              framealpha=1,
              borderpad=0.25)

    for fmt in (".png", ".svg"):
        figname = os.path.splitext(figname)[0] + fmt
        fig.savefig(figname, dpi=600)
        print(figname, "written.")

    return



if __name__ == "__main__":
    # results = special_plot_antarctica_and_peninsula_index()
    # for plot_type in ("index","extent"):
    # for plot_type in ("index",):
    #     fntemp=os.path.join(model_results_v2_5_plot_directory,"trends","R{0}_" + plot_type + "_trend.png")
    #     plot_time_series(region_number="all",
    #                       fname_template=fntemp,
    #                       melt_index_or_extent=plot_type,
    #                       include_trendline_only_if_significant=True,
    #                       offset_years_by_one=True,
    #                       include_trendline=False,
    #                       dpi=300,
    #                       extent_melt_days_threshold = 1,
    #                       verbose=True)
    special_plot_antarctica_and_all_regions()