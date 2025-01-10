import xarray as xr
import sys
import os
import multiprocessing
import warnings
import trend_analysis
sys.path.insert(0, '/home/561/mg5624/Aus_drought_trends/Aus_drought_trends/drought_trends/')
from MK_trends_functions import calculate_trendtest, find_season_data
warnings.filterwarnings("ignore", category=RuntimeWarning)


def calculate_block_variability(data, year_range, aggregate_over, season=None):
    """
    Calculates the variability for bloacks of aggregate_over periods.

    Args:
        data (xr.DataArray): data to find changin variability in
        year_range (list of str or int): years to calculate the blocks of variability over
        aggregate_over (int): number of years to aggregate over when finding the blocks of variability
        season (str): None or str of the season (e.g. 'DJF', etc.)

    Returns:
        blocked_variability (xr.DataArray): the rolling values of variability of data
    """
    if season != None:
        data = find_season_data(data, season, 'mean')

    year_from = year_range[0]
    year_to = year_range[-1]
    data = data.sel(time=slice(year_from, year_to))

    # year_block here assigns each timestep to it's year block e.g. when aggregate_over=5, for data starting in 1911, any data in the years 
    # 1911, 1912, 1913, 1914, and 1915 will be assigned to the 1911 year_block
    year_block = (data['time.year'].data - int(year_from)) // int(aggregate_over) * int(aggregate_over) + int(year_from)
    data.coords['year_block'] = ('time', year_block)

    # Group by 'year_block'
    ds_groups = data.groupby('year_block')

    # Find std dev for each year_block
    blocked_std_dev = ds_groups.std()
    blocked_std_dev = blocked_std_dev.rename({'year_block': 'time'})
    blocked_variability = blocked_std_dev
    
    return blocked_variability


def calculate_trend_of_variability(variable_name, year_range, aggregate_over, season=None, replace=True):
    """
    Calculates a trend in the rolling variability of data, or loads existing file if it already exists and replace=False.

    Args:
        variable_name (str): name of the variable to find the change in variability of
        year_range (list of str or int): years to calculate the rolling variability over
        aggregate_over (int): number of years to aggregate over when finding the rolling variability
        season (str): None or str of the season (e.g. 'DJF', etc.)
        replace (bool): if files already exist, choose to replace them, if False code loads exisiting file

    Returns:
        variability_trend (xr.DataArray): MK trend in the variability of the data
    """    
    file_in = trend_analysis.RAW_VAR_FILEPATHS[variable_name]
    year_from = year_range[0]
    year_to = year_range[-1]
    print(year_from, year_to)
    filepath_out = f'/g/data/w97/mg5624/RF_project/MK_test/predictor_variability/{variable_name}/MK_yue_wang/'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    filename_out = f'{year_from}-{year_to}_MK_yue_wang_MK_test_std_dev_{variable_name}.nc'
    if season is not None:
        filename_out = f'{season}_{filename_out}'

    if not os.path.exists(filepath_out + filename_out) or replace:
        data = xr.open_dataarray(file_in)
        variability_data = calculate_block_variability(data, year_range, aggregate_over, season=season)
        variability_trend = calculate_trendtest(variability_data, 'MK_yue_wang', year_from, year_to)
        variability_trend.to_netcdf(filepath_out + filename_out)
        print('file saved to:', filepath_out + filename_out)
    else:
        variability_trend = xr.open_dataset(filepath_out + filename_out)
    
    return variability_trend


VARS = [
    'ET',  
    'Precipitation', 
    'Runoff', 
    'Soil_Moisture'
]


def main():
    arguments = []
    for var in VARS:
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            arguments.append((var, ['1981', '2020'], 5, season))
   
    nb_cpus = 8
    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(calculate_trend_of_variability, arguments)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
