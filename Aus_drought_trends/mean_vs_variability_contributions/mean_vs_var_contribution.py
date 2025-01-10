import xarray as xr
import sys
sys.path.insert(0, '/g/data/w97/mg5624/RF_project/Aus_drought_trends/drought_trends/')
from MK_trends_function import regrid_mask


def find_mean_and_variability_contributions(drought_type, year_range):
    """
    Calculates the respective contributions to the drought trend from the mean and variability trend in the 
    driving variable (e.g., precipitation for meteorological drought). This is calculated using the equations:
      variability_contribution = (detrended variable drought trend / original drought trend) * 100
      mean_contribution = 100 - variability_contribution

    Args:
        drought_type (str): Type of drought, based on the driving variable (e.g., 'precip' for meteorological drought).
        year_range (list of str): List specifying the start and end years in the format [year_from, year_to].
        points_to_plot (str, optional): Specifies which points to plot. Options are 'all_points', 'increasing_drought', 
                                        'decreasing_drought', or 'mean_var_agree_direction'. Default is 'all_points'.
        plot (bool, optional): Whether to plot the data or not. Default is False.
        mask_non_signif (bool, optional): Masks out non-significant trends in the original drought trend in mean/variability 
                                          contribution plots. Default is False.
    """
    year_from = year_range[0]
    year_to = year_range[1]

    filepath_in = f'/g/data/w97/mg5624/RF_project/MK_test/drought_metrics/Aus/{drought_type}_percentile/MK_yue_wang/{year_from}-{year_to}/'
    detrended_drought_trend = xr.open_dataset(
        f'{filepath_in}/Aus_{year_from}-{year_to}_5_year_MK_yue_wang_test_{drought_type}_percentile_baseline_{year_from}_{year_to}_detrended.nc'
    )['MK_slope']

    mask = xr.open_dataarray(f'/g/data/w97/mg5624/RF_project/masks/regridded_awra_awap_mask.nc')
    mask = regrid_mask(mask, detrended_drought_trend)
    detrended_drought_trend = detrended_drought_trend.where(mask)
    
    original_drought_trend_ds = xr.open_dataset(
        f'{filepath_in}/Aus_{year_from}-{year_to}_5_year_MK_yue_wang_test_{drought_type}_percentile_baseline_{year_from}_{year_to}.nc'
    )

    original_drought_trend_ds = original_drought_trend_ds.where(mask)
    original_drought_trend = original_drought_trend_ds['MK_slope']

    variability_contribution = detrended_drought_trend / original_drought_trend * 100
    variability_contribution = variability_contribution.where(original_drought_trend != 0, 50) #when original trend is 0, contributions must be equal
    
    # Ensure all values between 0 and 100, anything above/below set to 100/0 respectively
    variability_contribution = variability_contribution.where(variability_contribution >= 0, 0).where(variability_contribution <= 100, 100)
    mean_contribution = 100 - variability_contribution

    filepath_out = f'/scratch/w97/mg5624/data/mean_var_contribution/{drought_type}_drought/'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)

    var_contribution_filename = f'{drought_type}_drought_variability_contribution_{year_from}_{year_to}.nc'
    mean_contribution_filename = f'{drought_type}_drought_mean_contribution_{year_from}_{year_to}.nc'
    mean_contribution.to_netcdf(filepath_out + mean_contribution_filename)
    variability_contribution.to_netcdf(filepath_out + var_contribution_filename)


DROUGHT_VARIABLES = [
    'precip',
    'soil_moisture',
    'runoff'
]


TIME_PERIODS = [
    ['1911', '2020'],
    ['1951', '2020'],
    ['1971', '2020'],
]


def main():
    for var in DROUGHT_VARIABLES:
        for period in TIME_PERIODS:
            find_mean_and_variability_contributions(var, period)  


if __name__ == "__main__":
    main()
