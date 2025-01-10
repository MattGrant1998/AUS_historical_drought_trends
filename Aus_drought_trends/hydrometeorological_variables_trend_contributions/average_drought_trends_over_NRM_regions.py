import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
import contributions_of_hydromet_vars_to_drought_trends as contributions_main
sys.path.insert(0, '/home/561/mg5624/Aus_drought_trends/Aus_drought_trends/drought_trends/')
import MK_trends_functions as MK_functions


DROUGHT_TYPES = [
    'Runoff',
    'Soil_Moisture'
]

SEASONS = [
    'DJF',
    'MAM',
    'JJA',
    'SON'
]


NRM_clusters = xr.open_dataset(
    '/g/data/w97/amu561/Steven_CABLE_runs/shapefiles/NRM/NRM_clusters.nc'
)['NRM_cluster']


def create_drought_timeseries_over_NRM(drought_variable, years, NRM_region):
    """
    Creates a timeseries of drought from the average over the specified NRM region.

    Args:
        drought_type (str): variable which the drought type is based off
        years (list of str): year range to have the timeseires over
        NRM_region (str): name of the NRM region to calculate drought metric over

    Returns:
        drought (xr.DataArray): timeseries of binary drought metric
    """
    drought_var_filepath = contributions_main.RAW_VAR_FILEPATHS[drought_variable]
    drought_var = xr.open_dataarray(drought_var_filepath).sel(time=slice(years[0], years[1]))
    drought_var_for_NRM_region = contributions_main.apply_NRM_mask(drought_var, NRM_region)
    average_drought_var_over_NRM_region = drought_var_for_NRM_region.mean(dim=['lat', 'lon'])
    percentiles = average_drought_var_over_NRM_region.groupby('time.month').reduce(np.percentile, q=15)
    drought = (average_drought_var_over_NRM_region.groupby('time.month') < percentiles).astype(int)
    drought = drought.drop_vars('month')
    drought = drought.rename('Drought')
    return drought


def find_MK_trend_in_NRM(drought_timeseries, years, season):
    """
    Calculates the MK trend over the average drought metric over the NRM region.

    Args:
        drought_timeseires (xr.DataArray): timeseries of binary drought metric
        years (list of str): year range to have the timeseires over

    Returns:
        drought_MK_slope (float): MK slope over the NRM region
        drought_MK_trend (int): MK trend over the NRM region
    """
    season_drought_data = MK_functions.find_season_data(drought_timeseries, season, 'sum')

    # Assign arbitrary lat and lon to data just so it works with MK trendtest function
    lats = [500]
    lons = [500]
    season_drought_data = season_drought_data.expand_dims({'lat': lats, 'lon': lons})

    MK_trend_da = MK_functions.calculate_trendtest(
        season_drought_data,
        'MK_yue_wang',
        years[0],
        years[1],
        events=True,
        aggregate=5
    )

    drought_MK_slope = MK_trend_da.MK_slope.values[0][0]
    drought_MK_trend = MK_trend_da.MK_trend.values[0][0]

    return drought_MK_slope, drought_MK_trend


def find_MK_trend_over_each_NRM(drought_type, years):
    """
    Finds the trend and slope of the drought data over each NRM region and season and saves it into a dataframe.

    Args:
        drought_type (str): name of the drought type's variable to find trend over
        years (list of str): year range to calculate trend over

    Returns:
        MK_trends (pd.DataFrame): the trend and slope of the drought over specified NRM region and season
    """
    MK_trends = pd.DataFrame()
    for NRM in contributions_main.NRM_REGIONS:
        drought_ts = create_drought_timeseries_over_NRM(drought_type, years, NRM)
        for season in SEASONS:
            MK_slope, MK_trend = find_MK_trend_in_NRM(drought_ts, years, season)
            MK_trends_dict = {
                'NRM_region': NRM,
                'Season': season,
                'MK_slope': MK_slope,
                'MK_trend': MK_trend
            }
            
            MK_trends_season = pd.DataFrame([MK_trends_dict])
            MK_trends = pd.concat((MK_trends, MK_trends_season), ignore_index=True)
            print('MK_TREND: ', MK_trends)

    filepath_out = f'/scratch/w97/mg5624/data/trend_analysis/NRM_mean_trends/{drought_type}_drought/{years[0]}-{years[1]}/'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    filename_out = f'NRM_mean_{drought_type}_drought_trends_{years[0]}-{years[0]}.csv'

    MK_trends.to_csv(filepath_out + filename_out)

    return MK_trends


def main():
    for drought_type in DROUGHT_TYPES:
        find_MK_trend_over_each_NRM(drought_type, ['1981', '2020'])


if __name__ == "__main__":
    main()
