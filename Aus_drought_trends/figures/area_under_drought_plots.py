import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pymannkendall as mk
import math


def load_area_under_drought_data(drought_metric_type, area, year_range=['1911', '2021']):
    """
    Loads spatial mean drought event data - needs to be multiplied by 100 to turn into area under drought (%).

    Args:
        drought_metric_type (str): name of the drought metric to load (precip_percentile, etc.)
        area (str): the area that the area under drought has been calculated over (name of NRM region)
        year_range (list of str): list of form ['start_year', 'end_year'] to constrain the data to

    Returns:
        data_constrained (xr.DataArray): area under drought data for area and metric specified, from 1911 to 2021
    """
    data_source = {
        'precip_percentile': 'AGCD',
        'runoff_percentile': 'AWRA',
        'soil_moisture_percentile': 'AWRA',
    }

    data_total_years = {
        'precip_percentile': '1900-2021',
        'runoff_percentile': '1911-2020',
        'soil_moisture_percentile': '1911-2020',
    }
    
    filepath = f'/g/data/w97/mg5624/RF_project/drought_metric/{drought_metric_type}/'
    filename = \
        f'{data_source[drought_metric_type]}_{drought_metric_type}' \
        f'_drought_metric_monthly_{data_total_years[drought_metric_type]}_{area}_spatial_mean.nc'
    spatial_mean = xr.open_dataarray(filepath + filename)
    area_under_drought = spatial_mean.sel(time=slice(year_range[0], year_range[-1])) * 100 #multiply by 100 to get %       
    return area_under_drought


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


def plot_area_under_drought_timseries(drought_metrics, area, year_range, y_axis_limits=None, trendlines='full_ts', labels=None):
    """
    Plots the percentage area under drought with the decadal confidence intervals and trendlines.

    Args:
        drought_metrics (list of str): list of the drought metrics to be plotted
        area (str): area that the area under drought is plotted for
        year_range (list of str): list of form ['start_year', 'end_year'] to constrain the data to
        y_axis_limits (array-like of float or int, optional): manual range for the y-axes
        trendlines (str or list, optional): what time period(s) to plot the trendline over, default 'full_ts', 
                                            else enter dates of form [[start_year1, end_year1], [start_year2, end_year2]]
        labels (list of str or None): titles of each plot (must be same length as drought_metrics)
    """
    drought_dataarrays = []
    for i, drought_type in enumerate(drought_metrics):
        drought_dataarrays.append(load_area_under_drought_data(drought_type, area, year_range))
    fig, axs = plt.subplots(1, len(drought_metrics), figsize=(9 * len(drought_metrics), 6), sharey=True)
    
    # Initialise min and max values
    min_y = float('inf')
    max_y = float('-inf')

    ts_color = 'dimgrey'
    trendline1_color = 'magenta'
    trendline2_color = 'red'
    decadal_mean_color = 'k'
    for i, dataarray in enumerate(drought_dataarrays):
        axs[i].set_title(labels[i])
        plotting_data = dataarray.rolling(time=60).mean()
        plotting_data.plot(ax=axs[i], label=labels[i], color=ts_color)
        if y_axis_limits == None:
            min_y = min(min_y, plotting_data.min().values) - 2 
            max_y = max(max_y, plotting_data.max().values) + 2
        
        first_start_year = int(year_range[0])
        last_end_year = int(year_range[0]) + (math.ceil((int(year_range[-1]) - int(year_range[0])) / 10) * 10) # if the year range isn't a perfect multiple of 10, this still works
        last_start_year = last_end_year - 10
        first_end_year = int(year_range[0]) + 10
        
        start_years = [str(year) for year in range(first_start_year, last_start_year + 1, 10)]
        end_years = [str(year) for year in range(first_end_year, last_end_year + 1, 10)]

        # Add in the trendline
        if trendlines == 'full_ts': 
            trendlines = [year_range]
        
        text_y_position = 0.95
        colors = [trendline1_color, trendline2_color]

        for c, trends in enumerate(trendlines):
            trend_data = plotting_data.sel(time=slice(str(trends[0]), str(trends[-1])))
            MK_result = mk.yue_wang_modification_test(trend_data.values)
            trend_line = np.arange(len(trend_data.values)) * MK_result.slope + MK_result.intercept
            p_value = "{:.2g}".format(MK_result.p)  # Round the p-value to 2 significant figures
            if float(p_value) < 0.001:
                signif = 'p < 0.001'
            elif float(p_value) < 0.05:
                signif = 'p < 0.05'
            else:
                signif = 'p = NS'
            axs[i].plot(trend_data.time, trend_line, label=f'Trend {c+1}', color=colors[c], linewidth=3.5)
            axs[i].text(0.02, text_y_position, signif, transform=axs[i].transAxes, fontsize=22, 
                    verticalalignment='top', color=colors[c])
            text_y_position -= 0.15

        # Plot decadal block averages
        for j in range(len(start_years)):
            start_year = start_years[j]
            end_year = end_years[j]
            block_avg = dataarray.sel(time=slice(start_year, end_year)).mean().values
            times = pd.date_range(start=start_year, end=end_year, freq='YE')
            block_avg_df = pd.DataFrame({'values': block_avg, 'time': times})
            block_avg_df.plot(x='time', y='values', ax=axs[i], color=decadal_mean_color, legend=False, label='10-year Mean', linewidth=2.5)

        # Plot mean, and decadal confidence intervals for the data
        baseline = dataarray.mean().values
        interval = decadal_confidence_intervals(dataarray, time=['1911', '1961'])
        positive_CI = baseline + interval
        negative_CI = baseline - interval
        axs[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
                
        x_start, x_end = axs[i].get_xlim()
        axs[i].fill_between(np.linspace(x_start, x_end, 500), negative_CI, positive_CI, 
                            color='lightblue', alpha=0.5, label='Confidence Interval')
        axs[i].set_xlim(x_start, x_end) 
        axs[i].set_xlabel('')
        axs[i].set_ylabel('Area Under Drought (%)')
        axs[i].set_ylabel('')
        if area == 'Wet_Tropics':
            axs[i].set_title(labels[i], fontsize=32, weight='bold', pad=20) # drought type label
        axs[i].tick_params(axis='both', which='both', labelsize=24) #tick labels
    
    area_name = area.replace('_', ' ')
    fig.suptitle(area_name, fontsize=34)

    # Adjust the layout
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
    for ax in axs:
        ax.set_ylim(min_y, max_y)

    figpath_out = '/scratch/w97/mg5624/plots/RF_project/results_analysis/stationarity_tests/area_under_drought/'
    if not os.path.exists(figpath_out):
        os.makedirs(figpath_out)

    names = '_'.join(labels)
    years = '_'.join(year_range)
    figname_out = f'{area}_{years}_area_under_drought_CIs_{names}.png'
    plt.savefig(figpath_out + figname_out)


NRM_REGIONS = [
    'Monsoonal_North',
    'Wet_Tropics',
    'Rangelands',
    'East_Coast',
    'Central_Slopes',
    'Murray_Basin',
    'S_SW_Flatlands',
    'Southern_Slopes'
]

METRICS = [
    'precip_percentile', 
    'soil_moisture_percentile', 
    'runoff_percentile'
]


def main():
    labels = ['Meteorological Drought', 'Agricultural Drought', 'Hydrological Drought']
    for area in NRM_REGIONS:
        plot_area_under_drought_timseries(METRICS, area, year_range=['1911', '2021'], trendlines=[['1911', '1965'], ['1965', '2021']], labels=labels, y_axis_limits=[0, 50])


if __name__ == "__main__":
    main()
    