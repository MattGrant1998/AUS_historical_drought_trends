import numpy as np
import pandas as pd
import xarray as xr
import os
import pymannkendall as mk
from sklearn.linear_model import LogisticRegression
import sys
import warnings
import multiprocessing
warnings.filterwarnings("ignore", category=RuntimeWarning) 


datadir = '/g/data/w97/mg5624/RF_project/'
plotdir = '/g/data/w97/mg5624/plots/RF_project/results_analysis/'
scratch = '/scratch/w97/mg5624/plots/RF_project/results_analysis/'


def regrid_mask(mask, grid):
    """
    Regrids a given mask to match the grid of another dataset.

    Parameters:
    mask (xarray.DataArray): The mask to be regridded. It should have latitude ('lat') as one of its dimensions.
    grid (xarray.DataArray): The target grid to which the mask will be regridded.

    Returns:
    xarray.DataArray: The regridded mask with the same dimensions as the input grid, converted to boolean type.
    """
    mask = mask.sortby('lat')
    mask = mask.astype(int)
    regridded_mask = mask.interp_like(grid, method='nearest')
    new_mask = regridded_mask.astype(bool)

    return new_mask


def aggregate_drought_events_per_time_period(drought_events_data, number_of_years, year_from, year_to, agg_type='sum'):
    """
    Finds the number of drought events per specified number of year for each grid cell.

    Args:
        drought_events_data (xr.DataArray): drought event data (monthly timescale)
        number_of_years (int): number of years to sum the number of droughts over
        year_from (int or str): year to take trend from
        year_to (int or str): year to take trend to
        agg_type (optional: 'mean' or 'sum'): whether to sum or mean over the time period

    Returns:
        drought_events_per_year (xr.DataArray): number of events per year
    """
    # This line resamples the data to be blocked sums over every 'number_of_years'. It starts from the first year of the data
    # and will end once all the years are included in one of the blocks. This could mean that the final block includes years 
    # for which we have no data for - the if conditions below sort this out
    # aggregated_metric = drought_events_data.resample(time=f'{number_of_years}Y', closed='right', label='left').sum(dim='time')

    # Sum number of drought events of blocks of the number of years specified. 
    # year_block here assigns each timestep to it's year block e.g. for data starting in 1911, any data in the years 
    # 1911, 1912, 1913, 1914, and 1915 will be assigned to the 1911 year_block to be summed together later
    year_block = (drought_events_data['time.year'].data - int(year_from)) // int(number_of_years) * int(number_of_years) + int(year_from)
    drought_events_data.coords['year_block'] = ('time', year_block)

    # Group by 'year_block' and sum over 'time'
    ds_groups = drought_events_data.groupby('year_block')
    if agg_type == 'sum':
        aggregated_metric = ds_groups.sum('time')
    elif agg_type == 'mean':
        aggregated_metric = ds_groups.mean('time')
    else:
        raise ValueError(f'aggregation type {agg_type} not valid for this function, use \'mean\' or \'sum\'.')
    aggregated_metric = aggregated_metric.rename({'year_block': 'time'})

    # Checks to ensure the block summing has worked as intended
    time = aggregated_metric['time'].values

    data_start_year = int(time[0])
    data_end_year = int(time[-1]) #converting to int gives it as years since 1970

    # These ensure we aren't including years in our blocked resampled data which we don't have any data for
    if data_start_year < int(year_from):
        aggregated_metric = aggregated_metric.isel(time=slice(1, None))
    # minus the one to number_of_years below, as year_blocks are closed on the right, e.g. 2016-2020 includes 2020
    if data_end_year > (int(year_to) - (number_of_years - 1)): 
        aggregated_metric = aggregated_metric.isel(time=slice(None, -1))
    return aggregated_metric


def find_logistic_regression_probability(data):
    """
    Calculates the trend slope of the data using a logistic regression model.

    Args:
        data (xr.DataArray): 3-D binary data 

    Returns:
        log_reg_prob (xr.DataArray): probability prediction from the logistic regression
    """
    log_reg_prob = data.copy()
    for i in data['lat'].values:
        for j in data['lon'].values:
            data_ij = data.sel(lat=i, lon=j).values
            # LogReg model only works when timeseries has at least two values, so if data is all 0 or 1, set slope to 0
            if all(x == data_ij[0] for x in data_ij):
                pass
            elif any(np.isnan(data_ij)):
                pass
            else:
                # create a 2-d array X with one column per input, and same number of rows as drought ts
                X = np.arange(len(data['time'])).reshape(-1, 1)
                log_reg_model = LogisticRegression(solver='saga', random_state=42, class_weight='balanced').fit(X, data_ij)
                prob_ij = log_reg_model.predict_proba(X)
                # print(prob_ij[:, 1])
                prob_ij = prob_ij[:, 1]
                log_reg_prob.loc[dict(lat=i, lon=j)] = prob_ij

    return log_reg_prob


def calculate_trendtest(data, test_type, year_from, year_to, events=False, aggregate=None):
    """
    Computes the MK or LogReg trendtest for each grid point in the data.

    Args:
        data (xr.DataArray): spatial and temporal data
        test_type (str): MK trend test type being performed or LogReg for logistic regression
        year_from (int): year from whcih the trend goes from
        year_to (int): year to which the trend goes to
        events (bool): if calculating for events set to True
        aggregate_years (int or str): if events=True, number of years to sum events over or 'LogReg' if using logistic regression

    Returns:
        MK_da (xr.DataArray): trend test result at each grid point
    """
    data = data.sel(time=slice(year_from, year_to))

    mask = xr.open_dataarray(f'{datadir}/masks/regridded_awra_awap_mask.nc')
    mask = regrid_mask(mask, data)
    data_masked = data.where(mask)

    if events:
        
        if isinstance(aggregate, int):
            data_masked = aggregate_drought_events_per_time_period(data_masked, aggregate, year_from, year_to)
        elif aggregate == 'LogReg':
            data_masked = find_logistic_regression_probability(data_masked)
        else:
            raise ValueError('aggregate_years must be specified when calculating MK trend of event data')
  
    trend_df = pd.DataFrame()
    print('Start calculating MK trend')
    for i in data_masked['lat'].values:
        for j in data_masked['lon'].values:
            data_ij = data_masked.sel(lat=i, lon=j).values
            # data_ij_df = data_ij.to_dataframe()
            # data_ij_df.reset_index(inplace=True)
            # if all values are nan, then skip this grid point
            if np.any(np.isnan(data_ij)):
                trend_dict = {'lat': i, 'lon': j, 'MK_trend': np.nan, 'MK_slope': np.nan}
                # break
            else:
                if test_type == 'MK_original':
                    trend_result = mk.original_test(data_ij)
                elif test_type == 'MK_hamed_rao':
                    trend_result = mk.hamed_rao_modification_test(data_ij)
                elif test_type == 'MK_yue_wang':
                    trend_result = mk.yue_wang_modification_test(data_ij)
                elif test_type == 'MK_seasonal_sens_slope':
                    trend_result = mk.seasonal_sens_slope(data_ij)
                elif test_type == 'LogReg':
                    trend_result = find_logistic_regression_probability(data_ij)
                if test_type == 'MK_seasonal_sens_slope':
                    trend_dict = {'lat': i, 'lon': j, 'MK_slope': trend_result.slope}
                else:
                    trend_dict = {'lat': i, 'lon': j, 'MK_trend': trend_result.trend, 'MK_slope': trend_result.slope}
            trend_df_ij = pd.DataFrame([trend_dict])
            trend_df = pd.concat((trend_df, trend_df_ij))

    if 'MK_trend' in trend_df.columns:
        trend_df['MK_trend'].replace({'no trend': 0, 'decreasing': -1, 'increasing': 1}, inplace=True)
    
    trend_df.set_index(['lat', 'lon'], inplace=True)
    trend_da = trend_df.to_xarray()
    
    return trend_da


def save_MK_test_df(data, var, year_from, year_to, season=None, test_type='original', events=False, aggregate_years=None):
    """
    Saves MK trend data as a NetCDF file.

    Args:
        data (xr.DataArray): Data to perform stats test on.
        var (str): Variable to calculate trend test for.
        year_from (int): Year trend goes from.
        year_to (int): Year trend goes to.
        season (str, optional): If requiring seasonal data then specify season (e.g. 'DJF'), else None.
        test_type (str, optional): Type of MK test that was conducted ('original', 'hamed_rao', 'yue_wang', 'seasonal_sens_slope', 'LogReg').
        events (bool, optional): If calculating for events set to True.
        aggregate_years (int or str, optional): If events=True, number of years to sum events over or 'LogReg' if using logistic regression.
    """
    if var in ['drought_events', 'drought_proba']:
        filepath = datadir + f'MK_test/RF_droughts/{var}/1911_model/{test_type}/'
    else:
        filepath = datadir + f'MK_test/RF_predictors/{var}/{test_type}/'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    da = calculate_trendtest(data, test_type, year_from, year_to, events, aggregate_years)

    if var in ['drought_events', 'drought_events_0.6', 'drought_events_0.7', 'drought_proba']:
        filename = f'{year_from}-{year_to}_{aggregate_years}_years_MK_{test_type}_test_{var}_1911_model.nc'
    else:
        filename = f'{year_from}-{year_to}__MK_{test_type}_test_{var}.nc'
    if season is not None:
        filename = f'{season}_{filename}'

    da.to_netcdf(filepath + filename)


def find_season_data(data, season, aggregate_method):
    """
    Creates an dataarray of all the points in the specified season.

    Args:
        data (xr.DataArray): data to create seasonal data of
        season (str): season of interest to pull from data
        aggregate_method (str): how to aggregate the season data ('sum' or 'mean')

    Returns:
        data_seas (xr.DataArray): data but only for the specified season
    """
    season_dict = {
        'DJF': 12,
        'MAM': 3,
        'JJA': 6,
        'SON': 9
    }

    if aggregate_method != None:
        if aggregate_method == 'sum':
            data_allseas = data.resample(time='QS-DEC').sum(dim='time').isel(time=slice(1, -1))
        elif aggregate_method == 'mean':
            data_allseas = data.resample(time='QS-DEC').mean(dim='time').isel(time=slice(1, -1))
        data_seas = data_allseas.sel(time=data_allseas['time.month'] == season_dict[season])
    else:
        season_group = data.groupby('time.season').groups[season]
        data_seas = data[season_group]

    return data_seas
