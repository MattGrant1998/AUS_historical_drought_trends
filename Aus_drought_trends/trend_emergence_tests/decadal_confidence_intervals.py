import numpy as np
import pandas as pd


def find_autocorrelation(timeseries, lag):
    """
    Finds the autocorrleation of a timeseries at the specified lag.

    Args:
        timeseries (xr.DataArray): timeseries of the some data, no spatial coordinates
        lag (int): the lag for the autocorrelation to be calculated over

    Returns:
        autocorrelation.values (float): the autocorrelation value of the timeseires for the specified lag
    """
    ts_mean = timeseries.mean(dim='time')

    # Calculate the denominator of the autocorrelation equation
    timeseries_anom_squared = (timeseries - ts_mean) ** 2
    denominator = timeseries_anom_squared.sum()

    # Calculate the numerator of the autocorrelation equation
    # lagged ts here plays the role of Y_i+k in autocorrelation equation 
    # (i.e. for any index ts value is Y_i, ts_lagged value is Y_i+k)
    if lag == 0:
        ts_lagged = timeseries
        ts = timeseries
    else:
        ts_lagged = timeseries.roll(time=-lag).isel(time=slice(None, -lag))
        ts = timeseries.isel(time=slice(None, -lag))
    
    ts_lagged_anom = ts_lagged - ts_mean
    ts_anom = ts - ts_mean
    product = ts_anom * ts_lagged_anom # product of the lagged anomaly and the non-lagged anomaly
    numerator = product.sum()

    # Final result
    autocorrelation = numerator / denominator

    return autocorrelation.values


def decadal_confidence_intervals(timeseries, time=None):
    """
    Calculates the 95% confidence intervals for the 10y means of the timeseries. This is to determine whether 
    changes in the means over successive decades can be considered statistically significant.

    Args:
        timeseries (xr.DataArray): timeseries of data
        time (list, optional): list containing the start and end times for slicing the timeseries. Defaults to None.

    Returns:
        positive_confidence (float): positive side of the confidence interval
    """
    if time is not None:
        timeseries = timeseries.sel(time=slice(time[0], time[-1]))
    variance = timeseries.var()
    autocorrelation_1 = find_autocorrelation(timeseries, 1)
    ts_size = timeseries.size
    ts_effective_size = ts_size * (1 - autocorrelation_1) / (1 + autocorrelation_1)
    positive_confidence = 1.96 * ((variance / ts_effective_size + variance / 10) ** 0.5)

    return positive_confidence.values
