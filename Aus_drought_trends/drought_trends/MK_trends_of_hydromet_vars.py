import xarray as xr
import warnings
import os
import multiprocessing
import MK_trends_functions as MK_functions
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def load_and_save_hydrometeorological_data(hydromet_variable, year_from, year_to, season=None, test_type='MK_yue_wang', replace=False):
    """
    Loads the data array of the hydrometeorological variable, calculates the MK trend test, and saves the results.

    Args:
        hydromet_variable (str): Name of the hydrometeorological variable to load.
        year_from (int): Year from which the trend analysis starts.
        year_to (int): Year to which the trend analysis ends.
        season (str, optional): If requiring seasonal data, specify the season (e.g., 'DJF'), else None.
        test_type (str, optional): Type of MK test to be conducted ('MK_original', 'MK_hamed_rao', 'MK_yue_wang', 'seasonal_sens_slope', 'LogReg').
        replace (bool, optional): If True, replace existing files.

    Returns:
        MK_data (xr.DataArray): MK trend test results for the specified variable.
    """
    data_files_in = {
        'ET': MK_functions.datadir + f'/ET_products/v3_6/ET/ET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc', 
        'Precipitation': MK_functions.datadir + '/Precipitation/AGCD/AGCD_v1_precip_total_r005_monthly_1900_2021.nc',
        'Runoff': MK_functions.datadir + f'/Runoff/AWRA/AWRAv7_Runoff_month_1911_2023.nc',
        'Soil_Moisture': MK_functions.datadir + f'AWRA_SM/root_zone_soil_moisture.nc'
    }
    
    hydromet_data = xr.open_dataarray(data_files_in[hydromet_variable])
    mask = xr.open_dataarray(f'{MK_functions.datadir}/masks/regridded_awra_awap_mask.nc')
    
    if hydromet_data.lat.equals(mask.lat):
        hydromet_data = hydromet_data.where(mask)
    else:
        mask = MK_functions.regrid_mask(mask, hydromet_data)
        hydromet_data = hydromet_data.where(mask)

    filepath_out = MK_functions.datadir + f'MK_test/hydromet_variables/{hydromet_variable}/{test_type}/'
    filename_out = f'{year_from}-{year_to}_{test_type}_MK_test_{hydromet_variable}.nc'

    if season != None:
        hydromet_data = MK_functions.find_season_data(hydromet_data, season, 'mean')
        filename_out = f'{season}_{filename_out}'
    else:
        hydromet_data = hydromet_data.groupby('time.year').sum(dim='time') # turn into accumulated annual values (one per year, not rolling)
        hydromet_data = hydromet_data.rename({'year': 'time'})

    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    
    if os.path.exists(filepath_out + filename_out) and not replace:
        print('passing')
        pass
    else:
        MK_data = MK_functions.calculate_trendtest(hydromet_data, test_type, year_from, year_to)
        MK_data.to_netcdf(filepath_out + filename_out)
    return MK_data


def run_MK_test_for_hydrometerological_variables():
    arguments = []                  
    start_year = '1981'
    end_year = '2020'
    VARIABLES = ['Precipitation', 'Runoff', 'ET', 'Soil_Moisture']
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        for var in VARIABLES:
            arguments.append((
                var, 
                'Aus',
                'MK', 
                start_year, 
                end_year, 
                season, 
                'MK_yue_wang', 
                True #replace
            ))

    nb_cpus = 3
    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(load_and_save_hydrometeorological_data, arguments)
    pool.close()
    pool.join()


def main():
    run_MK_test_for_hydrometerological_variables()


if __name__ == '__main__':
    main()
    