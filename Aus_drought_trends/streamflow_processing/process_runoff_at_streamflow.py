import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from rioxarray.exceptions import NoDataInBounds


def find_average_runoff_at_catchments(runoff, streamflow, catchment_shp, years):
    """
    For each of the stations found in streamflow, the average runoff is calculated for the catchment area.

    Args:
        runoff (xr.DataArray): gridded monthly runoff data across Australia
        streamflow (pd.DataFrame): monthly streamflow data at catchments
        catchment_shp (gpd.Shapefile): shapefiles for eah of the catchments
        years (list of str): years we want to save the runof data at catchments for

    Returns:
        runoff_at_catchments (pd.DataFrame): runoff data averaged over each of the catchment areas
    """
    year_from = years[0]
    year_to = years[1]

    runoff_time = runoff.time.dt.strftime('%Y-%m').values
    streamflow_time = streamflow['date'].values
    time_dates_match = np.array_equal(runoff_time, streamflow_time)

    if not time_dates_match:
        raise ValueError(
            f'Runoff dataarray covers a different time period ({runoff_time[0]}-{runoff_time[-1]}) '\
            f'to the streamflow dataframe ({streamflow_time[0]}-{streamflow_time[-1]})'
        )
    
    runoff_at_catchments = streamflow.copy()
    station_IDs = runoff_at_catchments.drop(columns=['date']).columns.tolist()
    crs = catchment_shp.crs

    runoff = runoff.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    runoff = runoff.rio.write_crs(crs)
    runoff = runoff.sel(time=slice(year_from, year_to))
    
    for station in station_IDs:
        shapefile = catchment_shp[catchment_shp['CatchID'] == station]
        try:
            runoff_at_station = runoff.rio.clip(shapefile.geometry, crs)
            runoff_mean_at_station = runoff_at_station.mean(dim=['lat', 'lon'])
            runoff_mean_at_station_df = runoff_mean_at_station.to_dataframe().drop(columns=['spatial_ref'])
            runoff_mean_at_station_df.index = runoff_mean_at_station_df.index.strftime('%Y-%m')
            runoff_mean_at_station_df.reset_index(inplace=True)
            runoff_at_catchments[station] = runoff_mean_at_station_df['Runoff']
        except NoDataInBounds:
            runoff_at_catchments[station] = np.nan
    return runoff_at_catchments


def save_runoff_at_catchments_df(years):
    """
    Calculates the average runoff over the streamflow catchments and saves the output as a csv.

    Args:
        years (list of str): years we want to save the runoff data at catchments

    Returns:
        None
    """
    runoff = xr.open_dataset('/g/data/w97/mg5624/RF_project/Runoff/AWRA/AWRAv7_Runoff_month_1911_2023.nc')['Runoff']
    runoff = runoff.sel(time=slice(years[0], years[1]))

    streamflow = pd.read_csv(f'/g/data/w97/mg5624/RF_project/Streamflow/processed/monthly_streamflow_data_processed_{years[0]}-{years[1]}_missing_cutoff_0.05.csv')
    shapefiles = gpd.read_file('/g/data/w97/mg5624/RF_project/Streamflow/02_location_boundary_area/shp/CAMELS_AUS_v2_Boundaries_adopted.shp')

    runoff_at_catchments = find_average_runoff_at_catchments(runoff, streamflow, shapefiles, years, drought_trend)
    filepath_out = '/g/data/w97/mg5624/RF_project/Streamflow/processed/'
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)
    filename = f'runoff_averaged_over_streamflow_catchments_{years[0]}-{years[1]}.csv'
    runoff_at_catchments.to_csv(filepath_out + filename, index=False)


def main():
    save_runoff_at_catchments_df(['1950', '2020'], drought_trend=False)

if __name__ == "__main__":
    main()
