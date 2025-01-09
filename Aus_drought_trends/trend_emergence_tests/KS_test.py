import numpy as np
import xarray as xr
import os
from scipy.stats import ks_2samp


def remove_nans(arr1, arr2):
    """
    Remove NaN values from two input arrays.

    Args:
        arr1 (np.array): The first input array.
        arr2 (np.array): The second input array.

    Returns:
        arr1, arr2 (tuple of np.arrays): Two arrays with NaN values removed.

    Notes:
        This function uses NumPy's isfinite function to identify and remove NaN values.
    """
    # Remove NaN values using NumPy's isfinite function
    arr1 = arr1[np.isfinite(arr1)]  # Remove NaN values from arr1
    arr2 = arr2[np.isfinite(arr2)]  # Remove NaN values from arr2

    return arr1, arr2  # Return the updated arrays


def return_ks_2samp(arr1, arr2):
    """
    Calculate the p-value for a given statistical test for two arrays.

    Args:
        arr1 (numpy array): The first input array.
        arr2 (numpy array): The second input array.

    Returns:
        ks_2samp(arr1, arr2).pvalue (float): The p-value of the specified statistical test.

    Notes:
        If either array contains only NaN values, the function returns NaN.
    """
    # Check if all values are nan
    if np.all(np.isnan(arr1)) or np.all(np.isnan(arr2)): return np.nan
    arr1, arr2 = remove_nans(arr1, arr2)

    return ks_2samp(arr1, arr2).pvalue


def find_ks_test_ds(dataarray, window, baseline):
    """
    Calculates the KS test across each grid cell in dataarray.

    Args:
        dataarray (xr.DataArray): spatial and temporal data
        window (int): window size for KS test
        baseline (list of str): list of form [start_baseline_year, end_baseline_year]

    Returns:
        ks_da (xr.DataArray): ks statistics for the dataarray
    """
    dataarray_window = (dataarray
                    .rolling(time=window, center=True, min_periods=window)
                    .construct('window_dim')
                    .persist()) 
    base_period_window = (dataarray.sel(time=slice(baseline[0], baseline[-1]))
                            .rename({'time':'window_dim'})
                            .persist())
    
    ks_ds = xr.apply_ufunc(
        return_ks_2samp,
        dataarray_window,
        base_period_window,
        input_core_dims=[['window_dim'], ['window_dim']],
        exclude_dims={'window_dim'},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

    ks_ds = ks_ds.compute()

    return ks_ds

data_source = {
    'precip': 'AGCD',
    'soil_moisture': 'AWRA',
    'runoff': 'AWRA'
}

years = {
    'precip': '1900-2021',
    'soil_moisture': '1911-2020',
    'runoff': '1911-2020'
}


def apply_ks_to_tud(drought_type, baseline):
    """
    Apply the Kolmogorov-Smirnov (KS) test to the time under drought (TUD) data for a given drought type and baseline period.

    Args:
        drought_type (str): The type of drought to analyze (e.g., 'meteorological', 'agricultural', etc.).
        baseline (list of int): The baseline period to compare against, specified as a list of two years [start_year, end_year].
    """
    annual_tud_filepath = f'/scratch/w97/mg5624/data/drought_metric/{drought_type}_percentile/'
    annual_tud_filename = f'{data_source[drought_type]}_{drought_type}_percentile_drought_metric_annual_{years[drought_type]}_baseline_1911_2020.nc'
    annual_tud = xr.open_dataarray(annual_tud_filepath + annual_tud_filename).sel(time=slice('1911', '2020'))
    ks_test_ds = find_ks_test_ds(annual_tud, 20, baseline)
    ks_test_filepath = '/scratch/w97/mg5624/data/time_of_emergence/ks_test/'
    ks_test_filename = f'ks_test_{drought_type}_time_under_drought_1911_2020_baseline_{baseline[0]}_{baseline[-1]}.nc'
    if not os.path.exists(ks_test_filepath):
        os.makedirs(ks_test_filepath)
    ks_test_ds.to_netcdf(ks_test_filepath + ks_test_filename)


def apply_ks_to_aud(drought_type, baseline):
    """
    Apply the Kolmogorov-Smirnov (KS) test to the area under drought (AUD) data for different regions in Australia.

    Args:
        drought_type (str): The type of drought to analyze (e.g., 'meteorological', 'agricultural', etc.).
        baseline (list of int): The baseline period to compare against, specified as a list of two years [start_year, end_year].

    Notes:
        This function processes data for multiple regions and saves the KS test results to NetCDF files.
    """
    for area in [
        'Central_Slopes',
        'East_Coast',
        'Murray_Basin',
        'Monsoonal_North',
        'Rangelands',
        'Southern_Slopes',
        'S_SW_Flatlands',
        'Wet_Tropics',
    ]:
        aud_filepath = f'/g/data/w97/mg5624/RF_project/drought_metric/{drought_type}_percentile/'
        aud_filename = f'{data_source[drought_type]}_{drought_type}_percentile_drought_metric_monthly_{years[drought_type]}_baseline_1911_2020_{area}_spatial_mean.nc'
        aud = xr.open_dataarray(aud_filepath + aud_filename)
        area_under_drought = aud.sel(time=slice('1911', '2020')).resample(time='YE').mean() * 100
        ks_test_ds = find_ks_test_ds(area_under_drought, 20, baseline)
        ks_test_filepath = f'/scratch/w97/mg5624/data/time_of_emergence/ks_test/aud/'
        ks_test_filename = f'{area}_ks_test_{drought_type}_area_under_drought_1911_2020_baseline_{baseline[0]}_{baseline[-1]}.nc'
        if not os.path.exists(ks_test_filepath):
            os.makedirs(ks_test_filepath)
        ks_test_ds.to_netcdf(ks_test_filepath + ks_test_filename)


DROUGHT_TYPES = [
    'precip', 
    'soil_moisture', 
    'runoff'
]


def main():
    for drought_type in DROUGHT_TYPES:
        apply_ks_to_aud(drought_type, ['1911', '1961'])
        

if __name__ == "__main__":
    main()
