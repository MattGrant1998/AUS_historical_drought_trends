import pandas as pd
import os
import MK_trends_functions as MK_functions


def find_drought_timeseries(catchment_data, years, var):
    """
    Finds a binary drought timeseries at each streamflow catchment for the variable provided, over the years specified.

    Args:
        data (pd.DataFrame): data to find drought timeseires of
        years (list of str): years to find droughts over (also used as baseline for metric)

    Returns:
        drought (pd.DataFrame): drought timeseries at each streamflow catchment
    """
    year_from = years[0]
    year_to = years[1]
    catchment_data['date'] = pd.to_datetime(catchment_data['date'], format='%Y-%m')

    catchment_data.set_index('date', inplace=True)
    catchment_data_rolling = catchment_data.rolling(window=3).mean()
    catchment_data_rolling.reset_index(inplace=True)
    
    catchment_data_years = catchment_data_rolling[
        (catchment_data_rolling['date'].dt.year >= int(year_from)) &
        (catchment_data_rolling['date'].dt.year <= int(year_to))
    ]
    
    catchment_stations = catchment_data_years.drop(columns=['date'])
    catchment_data_years['month'] = catchment_data_years['date'].dt.month
    catchment_drought = catchment_data_years.copy()
    drought_threshold = pd.DataFrame()
    for column in catchment_stations.columns:
        percentiles_15 = catchment_data_years.groupby('month')[column].quantile(0.15)
        catchment_drought[column] = catchment_data_years.apply(lambda row: 1 if row[column] < percentiles_15[row['month']] else 0, axis=1)
        drought_threshold[column] = percentiles_15

    filepath = f'/g/data/w97/mg5624/RF_project/drought_metric/streamflow_catchments/{var}_drought/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename_drought = f'{var}_drought_at_catchments_{year_from}-{year_to}_bymonth.csv'
    filename_percentiles = f'{var}_drought_threshold_at_catchments_{year_from}-{year_to}_bymonth.csv'

    drought_threshold.to_csv(filepath + filename_percentiles, index='CatchID', index_label='CatchID')
    catchment_drought.to_csv(filepath + filename_drought, index=False)

    return catchment_drought


def MK_trend_at_each_catchment(catchment_drought_data, years):
    """
    Finds the MK trend and sens slope of the drought data at the streamflow catchments.

    Args:
        catchment_drought_data (pd.DataFrame): binary drougt metric at streamflow catchments
        years (list of int): years to find droughts over (also used as baseline for metric)

    Returns:
        catchment_drought_trends (pd.DataFrame): drought trend and slope at each catchment
    """
    catchment_drought_data.rename(columns={'date': 'time'}, inplace=True)
    catchment_drought_trends = pd.DataFrame()
    catchment_drought_stations = catchment_drought_data.drop(columns='time')
    for station in catchment_drought_stations.columns:
        station_drought = catchment_drought_data[['time', station]]
        station_drought_da = station_drought.set_index('time').to_xarray()[station]
        
        # Assign arbitrary lat and lon to data just so it works with MK trendtest function
        lats = [500]
        lons = [500]
        station_drought_da = station_drought_da.expand_dims({'lat': lats, 'lon': lons})

        station_drought_MK = MK_functions.calculate_trendtest(
            station_drought_da,
            'MK_yue_wang',
            str(years[0]),
            str(years[1]),
            events=True,
            aggregate=5
        )

        station_drought_trend = station_drought_MK.MK_trend.values[0][0]
        station_drought_slope = station_drought_MK.MK_slope.values[0][0]

        station_trend_dict = {'CatchID': station, 'MK_trend': station_drought_trend, 'MK_slope': station_drought_slope}
        station_drought_df = pd.DataFrame([station_trend_dict])
        catchment_drought_trends = pd.concat((catchment_drought_trends, station_drought_df))

    return catchment_drought_trends


def create_dataframe_of_runoff_and_streamflow_drought_trends(years):
    """
    Creates a dataframe containing the runoff and streamflow drought metric MK trends and sens slopes

    Args:
        years (list of int): years to calculate the trend over
    """
    year_from = str(int(years[0]) - 1) #start it from one year before to ensure incorporation of all data when taking rolling averages later
    year_to = years[1]

    filepath_in = '/g/data/w97/mg5624/RF_project/Streamflow/processed/'
    runoff_at_catchments = pd.read_csv(filepath_in + f'runoff_averaged_over_streamflow_catchments_{year_from}-{year_to}.csv')
    runoff_droughts = find_drought_timeseries(runoff_at_catchments, years, 'runoff')
    runoff_drought_trends = MK_trend_at_each_catchment(runoff_droughts, years)
    runoff_drought_trends.rename(
        columns={
            'MK_trend': 'Runoff_MK_trend',
            'MK_slope': 'Runoff_MK_slope'
        }, 
        inplace=True
    )

    streamflow = pd.read_csv(filepath_in + f'monthly_streamflow_data_processed_{year_from}-{year_to}_missing_cutoff_0.05.csv')
    streamflow_droughts = find_drought_timeseries(streamflow, years, 'streamflow')
    streamflow_drought_trends = MK_trend_at_each_catchment(streamflow_droughts, years)
    streamflow_drought_trends.rename(
        columns={
            'MK_trend': 'Streamflow_MK_trend',
            'MK_slope': 'Streamflow_MK_slope'
        }, 
        inplace=True
    )

    combined_drought_trends = pd.merge(runoff_drought_trends, streamflow_drought_trends, on='CatchID')
    filepath_out = f'/g/data/w97/mg5624/RF_project/MK_test/streamflow_catchments/MK_yue_wang/{years[0]}-{years[1]}/'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    filename_out = f'runoff_streamflow_drought_MK_yue_wang_trend_test_{years[0]}-{years[1]}_bymonth.csv'

    combined_drought_trends.to_csv(filepath_out + filename_out, index=False)


def main():
    create_dataframe_of_runoff_and_streamflow_drought_trends(['1981', '2020'])
    create_dataframe_of_runoff_and_streamflow_drought_trends(['1951', '2020'])


if __name__ == "__main__":
    main()
