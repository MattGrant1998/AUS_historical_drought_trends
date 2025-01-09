import xarray as xr
import warnings
import MK_trends_functions as MK_functions
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def load_impact_drought_metric_data(season=None):
    """"
    Loads drought data.

    Args:
        season (str): if requiring seasonal data then specify season (e.g. 'DJF'), else 'None'

    Returns:
        data (xr.DataArray): data array of the required model and drought measure
    """
    file = MK_functions.datadir + f'drought_prediction/1911_model/drought_prediction_dataset_1911_model.nc'
    drought_ds = xr.open_dataset(file)

    if season == None:
        data = drought_ds.drought
    else:
        data = MK_functions.find_season_data(drought_ds.drought, season, 'sum')
        
    return data


def run_MK_test_for_drought_impact():
    YEARS = [
        ['1911', '2020'],
        ['1951', '2020'],
        ['1971', '2020']
    ]
    for years in YEARS:
        start_year = years[0]
        end_year = years[-1]
        MK_functions.save_MK_test_df(
            load_impact_drought_metric_data(
                season=None
            ), 
            'drought_events', start_year, end_year, season=None, test_type='MK_yue_wang', events=True, aggregate_years=5
        )


def main():
    run_MK_test_for_drought_impact()


if __name__ == '__main__':
    main()
