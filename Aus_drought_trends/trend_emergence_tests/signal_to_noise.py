import numpy as np
import xarray as xr
from scipy import signal
import statsmodels.api as sm
import os


lowess = sm.nonparametric.lowess


def apply_lowess(arr):
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) to a 1D array.

    Args:
        arr (numpy.ndarray): Input array containing the data to smooth. 
                             Can contain NaN values.

    Returns:
        yhat (np.ndarray): Smoothed array with the same shape as the input.
                           If all values in `arr` are NaN, the input array is returned unchanged.
    """
    # Check if the entire array consists of NaN values
    if all(np.isnan(arr)): return arr

    # Create an array of indices for the x-axis
    x = np.arange(arr.shape[0])
    
    # Apply the LOWESS smoothing function
    # `lowess` uses x as the independent variable and arr as the dependent variable
    yhat = lowess(arr, x, return_sorted=False)
    
    return yhat


def find_lowess_da(dataarray):
    """
    Fits a LOWESS model at each grid cell of the dataarray.

    Args:
        dataarray (xr.DataArray): spatial and temporal data

    Returns:
        lowess_da (xr.DataArray): dataarray of the lowess fits to the original data
    """
    dataarray = dataarray.chunk({'time':-1,
                                'lat':25,
                                'lon': 25})
    lowess_da = xr.apply_ufunc(
        apply_lowess, # The function to apply
        dataarray, # The data to apply the function to
        input_core_dims=[['time']], # What dimension to apply the funcion to
        output_core_dims=[['time']], # What dimension to be returned
        vectorize=True, # Needed to apply along time
        dask='parallelized', # We are using dask
        output_dtypes=[float] # Sometimes needed sometimes not
    )

    mask = xr.open_dataarray('/g/data/w97/mg5624/RF_project/masks/regridded_awra_awap_mask.nc')
    lowess_da = lowess_da.where(mask)
    
    lowess_da = lowess_da.compute()

    return lowess_da


def signal_to_noise(dataarray, baseline):
    """
    Calculates the signal to noise ratio over the whole period, based on the anomalies of the data from the specified baseline period.

    Args:
        dataarray (xr.DataArray): The input data array containing the time series data.
        baseline (tuple): A tuple containing the start and end years (inclusive) of the baseline period.

    Returns:
        S_to_N (xr.DataArray): The signal to noise ratio calculated over the entire period.

    Note:
    - If the data array contains latitude ('lat') coordinates, the function applies LOWESS to the data array directly.
    - If the data array does not contain latitude coordinates, the function applies LOWESS to the underlying numpy arrays.
    """
    baseline_period = dataarray.sel(time=slice(baseline[0], baseline[1])).mean()
    anomalies = dataarray - baseline_period

    detrended_data_anoms = xr.apply_ufunc(
        signal.detrend,
        anomalies,
        kwargs={'axis': 0},  # Detrend along the 'time' dimension
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True
    )

    if 'lat' in detrended_data_anoms.coords:
        detrended_lowess = find_lowess_da(detrended_data_anoms)
        lowess_da = find_lowess_da(anomalies)
    else:
        detrended_array = detrended_data_anoms.values
        array = anomalies.values
        detrended_lowess_arr = apply_lowess(detrended_array)
        lowess_arr = apply_lowess(array)
        detrended_lowess = detrended_data_anoms.copy(data=detrended_lowess_arr)
        lowess_da = anomalies.copy(data=lowess_arr)

    residuals = detrended_lowess - detrended_data_anoms
    noise = residuals.std()
    S_to_N = lowess_da / noise

    return S_to_N


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


def apply_s_to_n_to_tud(drought_type, baseline, test=False):
    """
    Apply signal-to-noise ratio calculation to time under drought (TUD) data.

    Args:
        drought_type (str): The type of drought metric to process (e.g., 'precip', 'soil_moisture', 'runoff').
        baseline (list): A list of two strings specifying the start and end years of the baseline period.
        test (bool, optional): If True, process a subset of the data for testing purposes. Default is False.
    """
    annual_tud_filepath = f'/scratch/w97/mg5624/data/drought_metric/{drought_type}_percentile/'
    annual_tud_filename = f'{data_source[drought_type]}_{drought_type}_percentile_drought_metric_annual_{years[drought_type]}_baseline_1911_2020.nc'
    annual_tud = xr.open_dataarray(annual_tud_filepath + annual_tud_filename).sel(time=slice('1911', '2020'))
    if test:
        annual_tud.sel(lat=slice(-34, -36), lon=slice(143, 145))
    s_to_n = signal_to_noise(annual_tud, baseline)
    s_to_n_filepath = f'/scratch/w97/mg5624/data/time_of_emergence/signal_to_noise/tud/'
    s_to_n_filename = f'signal_to_noise_{drought_type}_time_under_drought_1911_2020_baseline_{baseline[0]}_{baseline[-1]}.nc'
    if not os.path.exists(s_to_n_filepath):
        os.makedirs(s_to_n_filepath)
    s_to_n.to_netcdf(s_to_n_filepath + s_to_n_filename)


def apply_s_to_n_to_aud(drought_type, baseline):
    """
    Apply signal-to-noise ratio calculation to area under drought (AUD) data.

    Args:
        drought_type (str): The type of drought metric to process (e.g., 'precip', 'soil_moisture', 'runoff').
        baseline (list): A list of two strings specifying the start and end years of the baseline period.
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
        s_to_n = signal_to_noise(area_under_drought, baseline)
        s_to_n_filepath = f'/scratch/w97/mg5624/data/time_of_emergence/signal_to_noise/aud/'
        s_to_n_filename = f'{area}_signal_to_noise_{drought_type}_area_under_drought_1911_2020_baseline_{baseline[0]}_{baseline[-1]}.nc'
        if not os.path.exists(s_to_n_filepath):
            os.makedirs(s_to_n_filepath)
        s_to_n.to_netcdf(s_to_n_filepath + s_to_n_filename)


DROUGHT_TYPES = [
    'precip', 
    'soil_moisture', 
    'runoff'
]


def main():
    for drought_type in DROUGHT_TYPES:
        apply_s_to_n_to_tud(drought_type, ['1911', '1961'])
        apply_s_to_n_to_aud(drought_type, ['1911', '1961'])
        

if __name__ == "__main__":
    main()
