import xarray as xr
import os
import MK_trends_functions as MK_functions
import warnings
import multiprocessing
warnings.filterwarnings("ignore", category=RuntimeWarning) 


METRICS = [
    'precip_percentile',
    'runoff_percentile',
    'soil_moisture_percentile',
]

SOURCE = {
    'precip_percentile': 'AGCD',
    'runoff_percentile': 'AWRA',
    'soil_moisture_percentile': 'AWRA',
}


TOTAL_YEARS = {
    'precip_percentile': '1900-2021',
    'runoff_percentile': '1911-2020',
    'soil_moisture_percentile': '1911-2020',
}


TREND_YEARS = [
    ['1911', '2020'],
    ['1951', '2020'],
    ['1971', '2020'],
    ['1981', '2020'],
]


def calculate_MK_for_drought_metric(metric_name, test_type, aggregate_years, year_range, scale, characteristic='metric', baseline='1911_2020', season=None, detrended=False, replace=True):
    """
    Calculates MK test for specified drought metric.

    Args:
        metric_name (str): Name of the metric to calculate the trend of.
        test_type (str): Name of the trend test being done (usually 'MK_yue_wang').
        aggregate_years (int or str): Number of years to aggregate the events by, or 'LogReg' for logistic regression.
        year_range (list): List of form [year_from, year_to] for years to calculate trend between.
        scale (int): Number of months over which drought metrics were calculated.
        characteristic (str, optional): 'metric' or 'intensity', 'metric' works for drought events timing. Default is 'metric'.
        baseline (str, optional): Baseline period for the drought metric. Default is '1911_2020'.
        season (str, optional): Season to calculate trend of. Default is None, which calculates annual trends.
        detrended (bool, optional): If True, loads the detrended drought metric. Default is False.
        replace (bool, optional): If True, replaces existing files. Default is True.
    """
    year_from = year_range[0]
    year_to = year_range[-1]
    filepath_in = MK_functions.datadir + f'/drought_metric/{metric_name}/'
    filename_in = f'{SOURCE[metric_name]}_{metric_name}_drought_{characteristic}_monthly_1911-2020_baseline_{baseline}.nc'

    if scale != 3:
        filename_in = f'{filename_in[:-22]}_{scale}-month{filename_in[-22:]}'

    if detrended:
        filename_in = f'{filename_in[:-3]}_detrended.nc'

    # Check if the file exists arleady, will be skipped if replace is False and file exists
    filepath_out = MK_functions.datadir + f'/MK_test/drought_metrics/Aus/{metric_name}/{test_type}/{year_from}-{year_to}/'
    filename_out = f'Aus_{year_from}-{year_to}_{aggregate_years}_year_{test_type}_test_{metric_name}_baseline_{baseline}.nc'

    if characteristic == 'intensity':
        filename_out = f'Aus_{year_from}-{year_to}_{aggregate_years}_year_{test_type}_test_{metric_name}_intensity_baseline_{baseline}.nc'

    if season != None:
        filename_out = f'{season}_{filename_out}'

    if scale != 3:
        filename_out = f'{filename_out[:-3]}_{scale}-month.nc'
    
    if detrended:
        filename_out = f'{filename_out[:-3]}_detrended.nc'

    if os.path.exists(filepath_out + filename_out) and not replace:
        print(f'File {filepath_out + filename_out} already exists, so skipping')
        return filename_out + filepath_out
    
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)

    metric = xr.open_dataarray(filepath_in + filename_in)
    mask = xr.open_dataarray(f'{MK_functions.datadir}/masks/regridded_awra_awap_mask.nc')
    mask = MK_functions.regrid_mask(mask, metric)
    metric_masked = metric.where(mask)

    if season != None:
        if test_type == 'LogReg':
            metric_masked = MK_functions.find_season_data(metric_masked, season, None)
        else:
            metric_masked = MK_functions.find_season_data(metric_masked, season, 'sum')
    trend_df = MK_functions.calculate_trendtest(metric_masked, test_type, year_from, year_to, events=True, aggregate=aggregate_years)
    trend_df.to_netcdf(filepath_out + filename_out)


def main():
    arguments = []
    for metric in METRICS:
        for years in TREND_YEARS:
            for season in MK_functions.SEASONS:
                for agg in [2, 3, 7, 'LogReg']:
                    for detrended in [True, False]:
                        arguments.append((
                            metric, 
                            'MK_yue_wang', 
                            agg, 
                            years,
                            3,
                            'metric', #characteristic
                            '1911_2020', #baseline
                            season,
                            detrended, #detrended
                            False #replace
                        ))

    nb_cpus = 8
    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(calculate_MK_for_drought_metric, arguments)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
