import pandas as pd
import os

streamflow_stats = pd.read_csv('/g/data/w97/mg5624/RF_project/Streamflow/03_streamflow/streamflow_GaugingStats.csv')
streamflow_MLd = pd.read_csv('/g/data/w97/mg5624/RF_project/Streamflow/03_streamflow/streamflow_MLd_inclInfilled.csv')


def remove_stations_with_lots_of_missing_data(streamflow_data, streamflow_stats, proportion_limit):
    """
    Finds the stations where the proportion of missing data points is above proportion_limit and removes them from the streamflow data.

    Args:
        streamflow_data (pd.DataFrame): data with streamflow from all stations
        streamflow_stats (pd.DataFrame): data which includes column of the proportion of missing dat ain each station
        proportion_limit (float): cutof for acceptable missing data limit, removes anything above this level

    Returns:
        streamflow_filtered (pd.DataFrame): streamflow data with the stations of high missing datapoints removed
    """
    missing_data_stations = streamflow_stats[streamflow_stats['prop_missing_data'] >= proportion_limit]['station_id'].tolist()
    streamflow_filtered = streamflow_data.drop(columns=missing_data_stations)

    return streamflow_filtered


def remove_stations_not_covering_whole_time(streamflow_data, years):
    """
    Removes any stations which start after the years of interest start or end before the end.

    Args:
        streamflow_data (pd.DataFrame): data with streamflow from all stations
        years (list of int): range of years of interest of from [start_year, end_year]

    Returns:
        streamflow_cleaned_of_missing_years (pd.DataFrame): streamflow data only for stations that cover whole timeseries
    """
    streamflow_filtered_years = streamflow_data[
        (streamflow_data['year'] >= years[0]) &
        (streamflow_data['year'] <= years[1])
    ]
    missing_year_stations = streamflow_filtered_years.columns[(streamflow_filtered_years == -99.99).any()].tolist()
    streamflow_cleaned_of_missing_years = streamflow_filtered_years.drop(columns=missing_year_stations)
    
    return streamflow_cleaned_of_missing_years


def monthly_average_streamflow_data(daily_streamflow_data):
    """
    Takes the monthly avrage of the daily streamflow data

    Args:
        daily_streamflow_data (pd.DataFrame): daily streamflow data for stations

    Returns:
        monthly_streamflow_data (pd.DataFrame): monthly streamflow data for all stations in daily
    """
    daily_streamflow_data['date'] = pd.to_datetime(daily_streamflow_data[['year', 'month', 'day']])
    daily_streamflow_data.set_index('date', inplace=True)
    monthly_streamflow_data = daily_streamflow_data.resample('ME').mean()
    monthly_streamflow_data.index = monthly_streamflow_data.index.strftime('%Y-%m')
    monthly_streamflow_data.reset_index(inplace=True)
    monthly_streamflow_data.drop(columns=['year', 'month', 'day'], inplace=True)

    return monthly_streamflow_data


def save_processed_streamflow_data(processed_streamflow_data, years, missing_proportion_limit):
    """
    Just saves the streamflow data after processing it as necessary.

    Args:
        processed_streamflow_data (pd.DataFrame): fully processed streamflow data
        years (list of int): range of years of interest of from [start_year, end_year]
        missing_proportion_limit (float): cutof for acceptable missing data limit, removes anything above this level
    """
    filepath_out = '/g/data/w97/mg5624/RF_project/Streamflow/processed/'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    filename = f'monthly_streamflow_data_processed_{years[0]}-{years[1]}_missing_cutoff_{missing_proportion_limit}.csv'
    processed_streamflow_data.to_csv(filepath_out + filename, index=False)


def main():
    years = [1950, 2020]
    proportion_cutoff = 0.05
    save_processed_streamflow_data(
        monthly_average_streamflow_data(
            remove_stations_not_covering_whole_time(
                remove_stations_with_lots_of_missing_data(
                    streamflow_MLd,
                    streamflow_stats,
                    proportion_cutoff
                ),
                years
            )
        ),
        years,
        proportion_cutoff
    )


if __name__ == "__main__":
    main()
