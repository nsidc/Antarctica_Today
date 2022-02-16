"""Created on Sat Nov 21 08:48:43 2020.

@author: mmacferrin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import numpy
import os

from tb_file_data import antarctic_regions_dict, \
                         climatology_plots_directory, \
                         model_results_plot_directory
from melt_array_picklefile import read_model_array_picklefile
from compute_mean_climatology import _get_region_area_km2, \
                                     read_daily_melt_numbers_as_dataframe, \
                                     open_baseline_climatology_csv_as_dataframe

# Plot at a default 300 dpi
mpl.rcParams['figure.dpi'] = 300
# Plot colors, easy to adjust here.
PLOT_LIGHT_GREY = "#e8e8e8"
PLOT_MEDIUM_GREY = "#d0d0d0"
PLOT_BLUE = "#0000ff"
PLOT_RED = "#ff0000"
PLOT_LINEWIDTH = 1.5
PLOT_GRID_LINEWIDTH = 0.25
PLOT_GRID_LINECOLOR = "#b0b0b0"

def simple_plot_date_check(fname=os.path.join(model_results_plot_directory, "v3_years_coverage.png")):
    """See what dates are included in the dataset.

    Helpful for finding any missing data. Not useful for production, really, just for diagnostics.
    """
    melt_array, datetime_dict = read_model_array_picklefile()
    datetimes = list(datetime_dict.keys())

    fig, ax = plt.subplots(1,1)

    base_year = 1999

    for year in range(1979, 2020+1):
        start_date = datetime.datetime(year=year, month=10, day=1)
        end_date = datetime.datetime(year=year+1, month=4, day=30)

        dates_this_year = [dt for dt in datetimes if ((dt >= start_date) and (dt <= end_date))]
        dates_baseline_year = [datetime.datetime(year=(base_year if ((dt.month, dt.day) >= (start_date.month, start_date.day)) else base_year+1),
                                                 month=dt.month,
                                                 day=dt.day)

                               for dt in dates_this_year]

        year_arr = [year] * len(dates_baseline_year)

        ax.scatter(dates_baseline_year, year_arr, marker="|", s=5)


    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))

    loc = mpl.ticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    loc = mpl.ticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
    ax.yaxis.set_minor_locator(loc)

    # fig.autofmt_xdate()

    fig.tight_layout()

    if fname is None:
        plt.show()
    else:
        fig.savefig(fname, dpi=300)
        print (fname, "saved.")

def _add_plot_legend(ax, loc="upper center", adjust_ylim_range=True):
    """Add a legend to the plot."""
    # Make room for the legend at the top.
    if adjust_ylim_range:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1]*1.16)

    ax.legend(loc=loc, frameon=True, edgecolor="1.0", ncol=2, framealpha=1.0, borderaxespad=0.2, borderpad=0.1)

def _add_plot_title(ax, region_num, year, wrapyear=True):
    """Add a title to the plot."""
    ax.set_title("{0} Melt Extent {1}".format(antarctic_regions_dict[region_num],
                "{0} - {1}".format(year, year+1) if wrapyear else year))


def _add_plot_date_at_bottom(ax, date, x_fraction):
    """Add a small date at the bottom, indicating the last (current) day of the melt season."""
    # Align the label left, right, or center depending where the last data point is.
    if x_fraction < (1./14.):
        x_fraction = 0.0
        halign = "left"
    elif x_fraction > (13./14):
        x_fraction = 1.0
        halign = "right"
    else:
        halign = "center"

    ax.text(x_fraction, -0.10, date.strftime("%d %b %Y").lstrip("0"),
            verticalalignment="top",
            horizontalalignment=halign,
            transform=ax.transAxes,
            fontsize="x-small")

def _add_region_area_on_the_plot(fig,
                                 ax,
                                 region_number=0):
    """Add a small label on the axes with the area of the region involved.

    Ted felt like having the region area "in the plot area" made more sense than
    at the bottom. Test out putting it there instead.
    """
    region_name = antarctic_regions_dict[region_number]
    # The Amundsen Bellinghausen region name is to long, so wrap it:
    if region_number == 7:
        last_space_i = region_name.rfind(" ")
        if last_space_i > -1:
            region_name = region_name[0:last_space_i] + '\n' + region_name[last_space_i+1:]

    region_area = _get_region_area_km2(region_number)
    # Round to the nearest 1000 km2
    region_area_rounded = numpy.round(region_area, -3)

    region_text = "{0} Region:\n{1:,} km$^2$".format(region_name, region_area_rounded)

    ax.text(0.12,0.78,region_text,
            verticalalignment="top",
            horizontalalignment="left",
            transform=fig.transFigure,
            fontsize="medium",
            bbox=dict(boxstyle='square,pad=0', fc='white', ec='none'),
            alpha=1.0)


def _add_region_area_at_bottom(fig,
                               ax,
                               region_number=0):
    """Add a small label at the bottom-left with the area of the region involved."""
    region_name = antarctic_regions_dict[region_number]
    region_area = _get_region_area_km2(region_number)
    # Round to the nearest 1000 km2
    region_area_rounded = numpy.round(region_area, -3)

    region_text = "{0} Total Region Area: {1:,} km$^2$".format(region_name, region_area_rounded)

    ax.text(0.05,0.01,region_text,
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=fig.transFigure,
            fontsize="x-small")

def plot_current_year_melt_over_baseline_stats(current_date=None,
                                               region_num=0,
                                               doy_start = (10,1),
                                               doy_end = (4,30),
                                               outfile=None,
                                               gap_filled=True,
                                               dpi=300,
                                               verbose=True):
    """Read the melt data for the melt year up through the "current_datetime", and plot over the baseline climatology.

    current_datetime should be a date within the melt season (October 1 thru April 30).
    Other dates during the austral winter will simply default to ending April 30 of that year.

    Parameters
    ----------
    current_datetime: A datetime.datetime or datetime.time object giving the date within the melt season to plot up to.

    region_num: 0 thru 7. See tb_file_data.antarctic_regions_dict for details.

    doy_start: A 2-tuple (mm,dd) of the day-of-year start of the melt season. Default (10,1) (October 1)

    doy_end  : A 2-tuple (mm,dd) of the doy-of-year end of the melt season. Default (4,30) (April 30)
                NOTE: If current_datetime is during the winter, it will be changed to doy_end to only plot until the end of the melt year and not beyond.

    outfile:   Image file to write out.

    verbose:   Verbose output.

    Return
    ------
    None
    """
    df = read_daily_melt_numbers_as_dataframe(verbose=False, gap_filled=gap_filled)

    if current_date is None:
        current_date = df["date"].iloc[-1]

    current_doy = (current_date.month, current_date.day)

    # If current_doy is outside the melt season, just default to the last day of the melt season.
    if ((doy_start > doy_end) and (current_doy < doy_start) and (current_doy > doy_end)) or \
       ((doy_start <= doy_end) and ((current_doy < doy_start) or (current_doy > doy_end))):

           current_date = datetime.datetime(year=(current_date.year-1) if ((current_doy < doy_start) and (doy_start < doy_end)) else current_date.year,
                                            month=doy_end[0], day=doy_end[1])
           current_doy = doy_end

    # Convert the start date to a datetime object
    datetime_start = datetime.datetime(year=current_date.year if (current_doy > doy_start) else (current_date.year-1),
                                       month=doy_start[0], day=doy_start[1])

    datetime_end = datetime.datetime(year=datetime_start.year if doy_end > doy_start else (datetime_start.year + 1),
                                     month=doy_end[0], day=doy_end[1])

    records_in_range = df[(df.date >= datetime_start) & (df.date <= current_date)]
    datetimes = [dt for dt in records_in_range.date.apply(lambda x: x.to_pydatetime())]
    melt_pcts = records_in_range["R{0}_melt_fraction".format(region_num)]*100.

    # If there is no available for this year, just return.
    if len(datetimes) == 0:
        return

    # print("=========================================================")
    # print(datetime_start, current_date)
    # print(datetimes)
    # print(melt_pcts)
    # print(records_in_range)

    datetimes, melt_pcts = _add_nans_in_gaps(datetimes, melt_pcts, gap_days_max = 4)

    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(datetime_start, current_date)
    # print(datetimes)
    # print(melt_pcts)
    # print(records_in_range)

    if current_date > datetimes[-1]:
        current_date = datetimes[-1]

    fraction_x = float((current_date - datetime_start).days) / (datetime_end - datetime_start).days

    _plot_current_year_and_baseline(datetime_start,
                                   datetime_end,
                                   datetimes,
                                   melt_pcts,
                                   fraction_x,
                                   region_num=region_num,
                                   outfile=outfile,
                                   gap_filled=gap_filled,
                                   dpi=dpi,
                                   verbose=verbose)

    return

def _add_nans_in_gaps(datetimes,
                      melt_pcts,
                      gap_days_max=4):
    """Add nans to areas with large day gaps to make blanks in data series when plotted."""
    time_deltas = [(datetimes[i+1] - datetimes[i]) for i in range(len(datetimes)-1)]
    # print(time_deltas)

    gap_indices = numpy.where([td.days > gap_days_max for td in time_deltas])[0]
    # print(gap_indices)

    last_gap_index = 0
    new_datetimes = []
    new_melt_pcts = numpy.zeros((len(melt_pcts) + len(gap_indices)))
    last_melt_index = 0
    for gap_i in gap_indices:
        new_datetimes.extend(datetimes[last_gap_index:(gap_i+1)])
        new_datetimes.append(datetimes[gap_i] + (datetimes[gap_i+1] - datetimes[gap_i])/2)
        new_melt_pcts[last_melt_index:(last_melt_index+gap_i-last_gap_index+1)] = melt_pcts[last_gap_index:(gap_i+1)]

        last_melt_index = last_melt_index+gap_i-last_gap_index+1
        last_gap_index = gap_i+1

        new_melt_pcts[last_melt_index] = numpy.nan
        last_melt_index = last_melt_index + 1

    new_datetimes.extend(datetimes[last_gap_index:(len(datetimes))])
    new_melt_pcts[last_melt_index:] = melt_pcts[last_gap_index:]

    return new_datetimes, new_melt_pcts


def _plot_current_year_and_baseline(datetime_start,
                                   datetime_end,
                                   datetimes,
                                   current_year_percents,
                                   fraction_x,
                                   region_num=0,
                                   outfile=None,
                                   gap_filled=True,
                                   dpi=300,
                                   verbose=True):
    """Plot the current year's melt over the top of the baseline climatology.

    Climatology (in this case) defaults to fall 1980- spring 2011.

    Parameters
    ----------
    datetimes: A list of datetime.date or datetime.datetime objects of the year in question.
                Should only span the melt season (Oct 1 - Apri 30), or the plot will look funky.

    current_year_percents: A vector, equal in length to the datetimes array, giving
                the percentage of melt (fractino *100) covering that region each day of the melt year so far.

    region_num: 0 thru 7. See tb_file_data.antarctic_regions_dict for details.

    outfile:   Image file to write out.

    verbose:   Verbose output.

    Return
    ------
    None
    """
    if len(datetimes) == 0:
        return

    fig, ax = plt.subplots(1,1,tight_layout=True)

    # Plot the baseline climatology on the plot.
    _plot_baseline_climatology(region_num=region_num,
                              mpl_axes = ax,
                              return_axes=False,
                              outfile=None,
                              set_title=True,
                              current_year=datetime_start.year,
                              add_legend=False,
                              gap_filled=gap_filled,
                              verbose=False)

    plot_label = (str(datetimes[0].year) \
                  if (datetime_start.year == datetime_end.year)
                  else "{0}-{1}".format(datetime_start.year, str(datetime_end.year)[-2:])) + \
            " Melt Percentage"

    ax.plot(datetimes, current_year_percents, lw=PLOT_LINEWIDTH, color=PLOT_RED, ls="solid", label=plot_label)

    ax.set_ylim(ymin=0)

    _add_plot_legend(ax, loc="upper center", adjust_ylim_range=True)
    _add_plot_date_at_bottom(ax, datetimes[-1], x_fraction = fraction_x)
    # Put the region number at the bottom *if* it's not all of Antarctica.
    if region_num > 0:
        _add_region_area_on_the_plot(fig, ax, region_number=region_num)
        # _add_region_area_at_bottom(fig, ax, region_number=region_num)

    if outfile:
        if gap_filled and os.path.split(outfile)[1].find("gap_filled") == -1:
            base, ext = os.path.splitext(outfile)
            outfile = base + "_gap_filled" + ext

        if verbose:
            print("Plotting", outfile)
        fig.savefig(outfile, dpi=dpi)

    plt.close(fig)

def _plot_baseline_climatology(region_num=0,
                              mpl_axes=None,
                              baseline_start_year=1990,
                              baseline_end_year=2020,
                              return_axes=True,
                              outfile=None,
                              set_title=False,
                              current_year=2000,
                              add_legend=False,
                              gap_filled=True,
                              verbose=True):
    """Plot the baseline (median, inter-quartile, inter-decile) melt ranges into a matplotlib axis.

    Provides the opportunity to send the matplotlib.Axes instance as a parameter, so that
    other things can be plotted over it.

    Parameters
    ----------
    region_num: 0 thru 7. See tb_file_data.antarctic_regions_dict for details.

    mpl_axes: if a matplotlib.Axes instances is given, plot into that. Otherwise, create an axes instance.

    return_axes: return the matplotlib.Axes instance in which the figure was plotted.

    outfile: Plot/export the file to this output image filename.
            If outfile is provided, the mpl_axes argument will need to be provided
            as well so that we have the matplotlib.Figure instance as well.

    current_year: The starting year in which to plot/interpret these data.
            If we will be plotting a given year's data over this, use that year (i.e. 2020, for the 2020-21 melt season)
            Otherwise, it will default to using the year 2000-2001
            The "leap day" (Feb 29th) will be kept or removed appropriately, depending upon the year given.

    Return
    ------
    If "return_axes", return the matplotlib.Axes instance into which this graph was plotted.
    If "return_axes" is False or None, return None.
    """
    md_tuples, p10, p25, p50, p75, p90 = _get_baseline_percentiles_from_csv(region_number=region_num,
                                                                            gap_filled=gap_filled,
                                                                            verbose=False)
    # Convert to percentages
    p10 = p10 * 100.
    p25 = p25 * 100.
    p50 = p50 * 100.
    p75 = p75 * 100.
    p90 = p90 * 100.

    # Generate the datetimes. Must wrap the year over the new-year
    md_tuple_wrap_position = numpy.where([(md_tuples[i+1] < md_tuples[i]) for i in range(0,len(md_tuples)-1)])[0]
    if len(md_tuple_wrap_position) == 0:
        # It doesn't wrap, just use the same year for all days.
        datetimes = [datetime.date(current_year, md[0], md[1]) for md in md_tuples]
    else:
        assert len(md_tuple_wrap_position) == 1
        # Cut out Feb 29th if not a leap year.
        if ((2,29) in md_tuples):
            leap_day_i = md_tuples.index((2,29))
            if leap_day_i <= md_tuple_wrap_position[0]:
                try:
                    datetime.date(current_year, 2, 29)
                except ValueError:
                    md_tuples.remove((2,29))
                    p10 = numpy.append(p10[:leap_day_i], p10[(leap_day_i+1):])
                    p25 = numpy.append(p25[:leap_day_i], p25[(leap_day_i+1):])
                    p50 = numpy.append(p50[:leap_day_i], p50[(leap_day_i+1):])
                    p75 = numpy.append(p75[:leap_day_i], p75[(leap_day_i+1):])
                    p90 = numpy.append(p90[:leap_day_i], p90[(leap_day_i+1):])

            else:
                try:
                    datetime.date(current_year+1, 2, 29)
                except ValueError:
                    md_tuples.remove((2,29))
                    p10 = numpy.append(p10[:leap_day_i], p10[(leap_day_i+1):])
                    p25 = numpy.append(p25[:leap_day_i], p25[(leap_day_i+1):])
                    p50 = numpy.append(p50[:leap_day_i], p50[(leap_day_i+1):])
                    p75 = numpy.append(p75[:leap_day_i], p75[(leap_day_i+1):])
                    p90 = numpy.append(p90[:leap_day_i], p90[(leap_day_i+1):])

        datetimes = [datetime.date(current_year, md[0], md[1]) for md in md_tuples if (md >= md_tuples[0])] + \
                    [datetime.date(current_year+1, md[0], md[1]) for md in md_tuples if (md < md_tuples[0])]


    # Create the plotting axes
    if mpl_axes or outfile:
        ax = mpl_axes
        fig = None
    else:
        fig, ax = plt.subplots(1,1,tight_layout=True)

    ax.grid(axis='y', which='major', lw=PLOT_GRID_LINEWIDTH, color=PLOT_GRID_LINECOLOR)

    # Inter-decile range, 10-90%
    ax.fill_between(datetimes, p10, p90,color=PLOT_LIGHT_GREY,label="Interdecile Range")
    # Inter-quartile range, 25-75%
    ax.fill_between(datetimes, p25, p75,color=PLOT_MEDIUM_GREY,label="Interquartile Range")
    # Median (50%)
    baseline_label = "{0} - {1} Median".format(baseline_start_year, baseline_end_year)
    ax.plot(datetimes, p50, lw=PLOT_LINEWIDTH, color=PLOT_BLUE, ls="--", label=baseline_label)

    # Put ticks every month, month names in between.
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator()) # Tick every month.
    # Put a minor tick in the middle of each month, but don't display the tick
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=16))

    ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%b'))

    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))  ## Set major locators to integer values
    ax.set_xlim((datetimes[0], datetimes[-1]))

    ax.set_ylabel("Melt Extent (%)")

    ax.tick_params(axis="x", which="minor", labelsize="x-large")

    if set_title:
        _add_plot_title(ax, region_num, current_year, wrapyear=(len(md_tuple_wrap_position) > 0))

    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')

    if add_legend:
        _add_plot_legend(ax, adjust_ylim_range=True)

    if outfile:
        if verbose:
            print("Plotting", outfile)
        fig.savefig(outfile)

    if return_axes:
        return ax
    else:
        if mpl_axes is None:
            plt.close(fig)
        return

def _get_baseline_percentiles_from_csv(region_number=0,
                                       df=None,
                                       gap_filled=True,
                                       verbose=True):
    """Read the Antarctica Today baseline climatologies, return the (month,day) tuples and the 10,25,50,75,90th percentiles.

    Parameters
    ----------
    region_num: 0 thru 7. See tb_file_data.antarctic_regions_dict for details.
    df:  Pandas datafram containing the data. If None, open the dataframe and read from it.
        (Useful to open it only once and pass it along if we will be calling this fucntion repeatedlly.)
    verbose: Specifies whether to provide feedback (primarily if opening the CSV file.)

    Return
    ------
    6 return items, all of the same length
        - list of (month,day) tuples.
        - numpy array of 10th percentile values for each day, based on the baseline period.
        - numpy array of 25th percentile values for each day.
        - numpy array of 50th percentile values for each day (median).
        - numpy array of 75th percentile values for each day.
        - numpy array of 90th percentile values for each day.
    """
    if not df:
        df = open_baseline_climatology_csv_as_dataframe(gap_filled=gap_filled,
                                                        verbose=verbose)

    assert (0 <= region_number < len(antarctic_regions_dict))

    md_tuples = [md for md in zip(df.month, df.day)]
    p10 = df["R{0}_fraction_10".format(region_number)]
    p25 = df["R{0}_fraction_25".format(region_number)]
    p50 = df["R{0}_fraction_50".format(region_number)]
    p75 = df["R{0}_fraction_75".format(region_number)]
    p90 = df["R{0}_fraction_90".format(region_number)]

    return md_tuples, p10, p25, p50, p75, p90


def DO_IT_ALL(gap_filled=True):
    """Do it all."""
    simple_plot_date_check()

    for year in range(1979,2021):
        for region in range(0,8):
            fname = os.path.join(climatology_plots_directory, "R{0}_{1}-{2}.png".format(region, year, year+1))
            plot_current_year_melt_over_baseline_stats(datetime.datetime(year=year+1, month=4, day=30),
                                                       region_num=region,
                                                       outfile = fname,
                                                       gap_filled=gap_filled)


if __name__ == "__main__":

    # for gap_filled in (False, True):
    #     DO_IT_ALL(gap_filled=gap_filled)

    # df = open_baseline_climatology_csv_as_dataframe()

    # for i in range(len(antarctic_regions_dict)):
        # plot_baseline_climatology(region_num = i, set_title = True, add_legend = True)

    # save_daily_melt_numbers_to_csv()

    # for year in range(2019,2020):
    #     for region in range(0,8):
    year=2020
    # region=0
    for region_num in range(8):
        for ext in ("png", "svg", "eps"):
            fname = os.path.join(climatology_plots_directory, "R{0}_{1}-{2}.{3}".format(region_num, year, year+1, ext))
            plot_current_year_melt_over_baseline_stats(current_date= datetime.datetime(year+1,4,30),
                                                       region_num=region_num,
                                                       outfile = fname,
                                                       dpi=600,
                                                       gap_filled=True)

    # df = plot_current_year_melt_over_baseline_stats(datetime.datetime(2020,6,30))

    # simple_plot_date_check()
