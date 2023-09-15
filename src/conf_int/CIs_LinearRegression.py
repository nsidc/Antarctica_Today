#!/usr/bin/env python

'''
Calculating confidence intervals for a linear regression

Heavily inspired (read: copied) from:
    linfit.py - example of confidence limit calculation for linear
                regression fitting.

    http://tomholderness.wordpress.com/2013/01/10/confidence_intervals/

# References:
# - Statistics in Geography by David Ebdon (ISBN: 978-0631136880)
# - Reliability Engineering Resource Website:
# - http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm
# - University of Glascow, Department of Statistics:
# - http://www.stats.gla.ac.uk/steps/glossary/confidence_intervals.html#conflim

By Kirstie Whitaker, on 27th September 2013
    Contact:  kw401@cam.ac.uk
    GitHubID: HappyPenguin

'''

# ====== IMPORTS =============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
# ============================================================================

# ====== FUNCTIONS ===========================================================
def lin_fit(x,y):
    '''
    Predicts the values for a best fit between numpy arrays x and y

    Parameters
    ----------
    x: 1D numpy array
    y: 1D numpy array (same length as x)

    Returns
    -------
    p:     parameters for linear fit of x to y
    y_err: 1D array of difference between y and fit values
               (same length as x)

    '''

    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    fit = p(x)

    y_err = y - fit

    return p, y_err

# ----------------------------------------------------------------------------

def conf_calc(x, y_err, c_limit=0.975, test_n=50):
    '''
    Calculates confidence interval of regression between x and y

    Parameters
    ----------
    x:       1D numpy array
    y_err:   1D numpy array of residuals (y - fit)
    c_limit: (optional) float number representing the area to the left
             of the critical value in the t-statistic table
             eg: for a 2 tailed 95% confidence interval (the default)
                    c_limit = 0.975
    test_n:  (optional) integer representing number of equally-spaced
             x-points to predict in the range min(x) to max(x)

    Returns
    -------
    p_x:   1D array, length "test_n", with values spread evenly between
           min(x) and max(x)
    confs: 1D numpy array of predicted y values for x inputs

    '''
    # Define the variables you need
    # to calculate the confidence interval
    mean_x = np.mean(x)			# mean of x
    n = len(x)				# number of samples in original fit
    tstat = t.ppf(c_limit, n-1)         # appropriate t value
    s_err = np.sum(np.power(y_err,2))	# sum of the squares of the residuals

    # create series of new test x-values to predict for
    p_x = np.linspace(np.min(x),np.max(x),test_n)

    confs = tstat * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
			((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))

    return p_x, confs

# ----------------------------------------------------------------------------

def ylines_calc(p_x, confs, fit):
    '''
    Calculates the three lines that will be plotted

    Parameters
    ----------
    p_x:   1D array with values spread evenly between min(x) and max(x)
    confs: 1D array with confidence values for each value of p_x
    fit:   Predicted x -> y function (typically derived from np.poly1d)

    Returns
    -------
    p_y:    1D array with values corresponding to fit line (for p_x values)
    upper:  1D array, values corresponding to upper confidence limit line
    lower:  1D array, values corresponding to lower confidence limit line

    '''
    # now predict y based on test x-values
    p_y = fit(p_x)

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    return p_y, lower, upper

# ----------------------------------------------------------------------------

def plot_linreg_CIs(x, y, p_x, p_y, lower, upper):

    # set-up the plot
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Linear regression and confidence limits')

    # plot sample data
    plt.plot(x, y, 'bo', label='Sample observations')

    # plot line of best fit
    plt.plot(p_x, p_y, 'r-', label='Regression line')

    # plot confidence limits
    plt.plot(p_x, lower, 'b--', label='Lower confidence limit (95%)')
    plt.plot(p_x, upper, 'b--', label='Upper confidence limit (95%)')

    # show the plot
    plt.show()


def confidence_bounds(x, y, c_limit=0.975, test_n=50):
    """
    Calculates the predicted (trend) line, and lower & upper confidene bounds
    from original x,y measurements

    Parameters
    ----------
    x: 1D numpy array
    y: 1D numpy array (same length as x)
    c_limit: (optional) float number representing the area to the left
             of the critical value in the t-statistic table
             eg: for a 2 tailed 95% confidence interval (the default)
                    c_limit = 0.975
    test_n:  (optional) integer representing number of equally-spaced
             x-points to predict in the range min(x) to max(x)

    Returns
    -------
    p_x:    1D array with values spread evenly between min(x) and max(x)
            (length test_n)
    p_y:    1D array with values corresponding to fit line (for p_x values)
            (length test_n)
    upper:  1D array, values corresponding to upper confidence limit line
    lower:  1D array, values corresponding to lower confidence limit line

    """
    # Fit x to y
    p, y_err = lin_fit(x ,y)

    # Calculate confidence intervals
    p_x, confs = conf_calc(x, y_err, c_limit=c_limit, test_n=test_n)

    # Calculate the lines for plotting:
    # The fit line, and lower and upper confidence bounds
    p_y, lower, upper = ylines_calc(p_x, confs, p)

    return p_x, p_y, lower, upper

# ============================================================================

# ====== MAIN CODE, SAMPLE USE ===============================================
if __name__ == "__main__":
    # Define example data
    x = np.linspace(1,15,50)
    y = x * 4 + 2.5
    x = x + np.random.random_sample(size=x.shape) * 20
    y = y + np.random.random_sample(size=x.shape) * 20

    # Fit x to y
    p, y_err = lin_fit(x, y)

    # Calculate confidence intervals
    p_x, confs = conf_calc(x, y_err, 0.975)

    # Calculate the lines for plotting:
    # The fit line, and lower and upper confidence bounds
    p_y, lower, upper = ylines_calc(p_x, confs, p)

    # Plot these lines
    plot_linreg_CIs(x, y, p_x, p_y, lower, upper)

# ============================================================================
