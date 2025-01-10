import pandas as pd 
import numpy as np 
import xarray as xr
import os
import multiprocessing
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


datadir = '/g/data/w97/mg5624/RF_project/'


RAW_VAR_FILEPATHS = {
    'ET': '/g/data/w97/mg5624/RF_project/ET_products/v3_6/ET/ET_1980-2021_GLEAM_v3.6a_MO_Australia_0.05grid.nc',
    'Precipitation': '/g/data/w97/mg5624/RF_project/Precipitation/AGCD/AGCD_v1_precip_total_r005_monthly_1900_2021.nc',
    'Soil_Moisture': '/g/data/w97/mg5624/RF_project/AWRA_SM/root_zone_soil_moisture.nc',
    'Runoff': '/g/data/w97/mg5624/RF_project/Runoff/AWRA/AWRAv7_Runoff_month_1911_2023.nc'
}


def apply_NRM_mask(data, NRM_region):
    """
    Applies the NRM mask to data for the specified region

    Args:
        data (xr.DataArray or xr.Dataset): data to be masked to NRM region
        NRM_region (str): name of NRM region to mask data to

    Returns:
        data_NRM_masked (xr.DataArray or xr.Dataset): data masked to the specified NRM region
    """
    NRM_clusters = xr.open_dataset(
        '/g/data/w97/amu561/Steven_CABLE_runs/shapefiles/NRM/NRM_clusters.nc'
    )['NRM_cluster']

    NRM_REGIONS = {
        'Central_Slopes': 1,
        'East_Coast': 2,
        'Murray_Basin': 4,
        'Monsoonal_North': 5,
        'Rangelands': 6,
        'Southern_Slopes': 7,
        'S_SW_Flatlands': 8,
        'Wet_Tropics': 9,
    }

    NRM_clusters_regrid = NRM_clusters.interp_like(data, method='nearest')
    data_NRM_masked = data.where(NRM_clusters_regrid == NRM_REGIONS[NRM_region])

    return data_NRM_masked


def create_trend_df(
        drought_type, 
        years, 
        predictors, 
        region,
        season
):
    """
    Creates a dataframe of the trends for each grid in the specified NRM_region, for the drought type and predictors specified.

    Args:
        drought_type (str): drought type for which we want to predict the trend of
        years (str): year range the trend is taken over ('1981-2020' or '1911-2020')
        predictors (list of str): name of the predictors for which the trend will be used to predict drought trend
        region (str): name of the NRM region to predict trends for
        season (st): if conducting seasonal analysis specify season (DJF, etc.), if None conducts analysis on annual trend 
    
    Returns:
        trends_df (pd.DataFrame): the trends for the drought and each pedictor at each grid of the NRM region
    """
    drought_filepath = f'/g/data/w97/mg5624/RF_project/MK_test/drought_metrics/Aus/{drought_type}_percentile/MK_yue_wang/{years}/'
    drought_filename = f'Aus_{years}_5_year_MK_yue_wang_test_{drought_type}_percentile.nc'
    if season is not None:
        drought_filename = f'{season}_{drought_filename}'
    drought_trend = xr.open_dataset(drought_filepath + drought_filename)[f'MK_slope']
    drought_trend = drought_trend.rename(f'{drought_type}_drought_trend')
    drought_trend_NRM = apply_NRM_mask(drought_trend, region)
    trends = drought_trend_NRM.to_dataset()

    for predictor in predictors:
        # Add mean trend
        predictor_mean_trend_filepath = f'/g/data/w97/mg5624/RF_project/MK_test/RF_predictors/{predictor}/MK_yue_wang/'
        predictor_mean_trend_filename = f'1981-2020_MK_yue_wang_MK_test_{predictor}.nc'
        if season is not None:
            predictor_mean_trend_filename = f'{season}_{predictor_mean_trend_filename}'
        predictor_mean_trend = xr.open_dataset(predictor_mean_trend_filepath + predictor_mean_trend_filename)['MK_slope']
        predictor_mean_trend_NRM = apply_NRM_mask(predictor_mean_trend, region)
        predictor_mean_trend_NRM = predictor_mean_trend_NRM.rename(f'{predictor}_mean_trend')
        trends = trends.merge(predictor_mean_trend_NRM)

        # Add variability trend
        predictor_variability_trend_filepath = f'/g/data/w97/mg5624/RF_project/MK_test/predictor_variability/{predictor}/MK_yue_wang/'
        predictor_variability_trend_filename = f'1981-2020_MK_yue_wang_MK_test_std_dev_{predictor}.nc'
        if season is not None:
            predictor_variability_trend_filename = f'{season}_{predictor_variability_trend_filename}'
        predictor_variability_trend = xr.open_dataset(predictor_variability_trend_filepath + predictor_variability_trend_filename)['MK_slope']
        predictor_variability_trend_NRM = apply_NRM_mask(predictor_variability_trend, region)
        predictor_variability_trend_NRM = predictor_variability_trend_NRM.rename(f'{predictor}_std_dev_trend')
        trends = trends.merge(predictor_variability_trend_NRM)
      
    trends_df = trends.to_dataframe().reset_index()
    trends_df.dropna(inplace=True)
    if 'spatial_ref' in trends_df.columns:
        trends_df.drop('spatial_ref', axis=1, inplace=True)

    return trends_df


def train_single_RF_model(
        target_training, 
        predictors_training,
        random_state
):
    """
    Trains single RF model for defined random state.

    Args:
        target (pd.Series or np.array): target variable to train model on

        region (str): name of the NRM region to predict trends for
        random_state (int): the random state of the model
        predict (str: 'slope' or 'trend'): whether to predict the MK trend or the sens slope magnitude

    Returns:
        model (sklearn.model): trained statistical model
    """
    model = RandomForestRegressor(
        n_estimators=500, random_state=random_state
    )
    model.fit(predictors_training, target_training)
    return model


def test_trend_prediction_model(
        drought_type, 
        years, 
        predictors, 
        region, 
        n_iterations,
        season
):
    """
    Trains n_iterated RF models and tests skill at predicting the drought trends on withheld trend data.

    Args:
        drought_type (str): drought type for which we want to predict the trend of
        years (str): year range the trend is taken over ('1981-2020' or '1911-2020')
        predictors (list of str): name of the predictors for which the trend will be used to predict drought trend
        region (str): name of the NRM region to predict trends for
        n_iterations (int): how many times to iterated the models
        season (str): if conducting seasonal analysis specify season (DJF, etc.), if None conducts analysis on annual trend
    Returns:
        performance_scores (pd.DataFrame): performance scores to assess model's skill
    """
    trends_df = create_trend_df(drought_type, years, predictors, region, season)
    predictors = trends_df.drop([f'{drought_type}_drought_trend', 'lat', 'lon'], axis= 1) 
    target = trends_df[f'{drought_type}_drought_trend']
    performance_scores = pd.DataFrame(columns=['Mean Squared Error', 'Mean Absolute Error'])

    for seed in range(n_iterations):
        predictors_train, predictors_test, target_train, target_test = train_test_split( 
            predictors, target, test_size=0.3, random_state=seed
        )
        model = train_single_RF_model(
            target_train, predictors_train,seed
        )
        test_predictions = model.predict(predictors_test)        
        performance_scores.loc[seed] = [
            mean_squared_error(target_test, test_predictions),
            mean_absolute_error(target_test, test_predictions)
        ]
    
    return performance_scores


def find_variable_importance_for_n_iterated_models(
        drought_type, 
        years, 
        predictors, 
        region, 
        n_iterations,
        season,
        add_random=True
):
    """
    Finds the variable imporance for n iterated models.

    Args:
        drought_type (str): drought type for which we want to predict the trend of
        years (str): year range the trend is taken over ('1981-2020' or '1911-2020')
        predictors (list of str): name of the predictors for which the trend will be used to predict drought trend
        region (str): name of the NRM region to predict trends for
        n_iterations (int): how many times to iterated the models
        season (st, optional): if conducting seasonal analysis specify season (DJF, etc.), if None conducts analysis on annual trend
        add_random (bool): if True, adds a column of randomly generated data to the trends_df 

    Returns:
        variable_importance_df (pd.DataFrame): varibale importance scores for each model
    """
    trends_df = create_trend_df(
        drought_type, years, predictors, region, season
    )
    predictors = trends_df.drop([f'{drought_type}_drought_trend', 'lat', 'lon'], axis= 1) 
    target = trends_df[f'{drought_type}_drought_trend']

    if add_random:
        variable_importance = np.zeros((len(predictors.columns) + 1, n_iterations))
    else:
        variable_importance = np.zeros((len(predictors.columns), n_iterations))

    for seed in range(n_iterations):
        if add_random:
            # Create random variable which consists of random values from -100 to 100
            predictors['Random_variable'] = np.random.rand(len(trends_df)) * 200 - 100
        model = train_single_RF_model(
            target, predictors, seed
        )
        variable_importance[:, seed] = model.feature_importances_

    variable_importance_df = pd.DataFrame(variable_importance.T, columns=predictors.columns)
    return variable_importance_df


def save_performance_metric_and_variable_importance_scores(
        drought_type, 
        years, 
        predictors, 
        region, 
        stats_model_name,
        season,
        add_random=False
):
    """
    Trains multiple models, evaluates their performance, and saves the results.

    Args:
        drought_type (str): The type of drought for which the trend prediction is being performed.
        years (str): The range of years over which the trend is calculated (e.g., '1981-2020' or '1911-2020').
        predictors (list of str): List of predictor variables used for trend prediction.
        region (str): The NRM region for which the trends are being predicted.
        stats_model_name (str): The name of the statistical model being used.
        season (str): The season for which the analysis is conducted (e.g., 'DJF', 'MAM', 'JJA', 'SON'). If None, the analysis is conducted on annual trends.
        add_random (bool): If True, adds a column of randomly generated data to the trends_df for variable importance analysis.
    """
    perfomance_metrics = test_trend_prediction_model(
            drought_type, 
            years,
            predictors, 
            region, 
            stats_model_name, 
            100,
            season
    )
    
    var_import = find_variable_importance_for_n_iterated_models(
            drought_type, 
            years, 
            predictors, 
            region, 
            100,
            season,
            add_random=add_random
    )

    var_import_df_filepath = f'/scratch/w97/mg5624/data/trend_analysis/variable_importance/{drought_type}_drought/{season}/Random_Forest_Regression/'
    var_import_df_filename = f'{season}_{region}_{drought_type}_drought_{years}_trend_analysis_Random_Forest_Regression_std_dev.csv'
    if not os.path.exists(var_import_df_filepath):
        os.makedirs(var_import_df_filepath)
    var_import.to_csv(var_import_df_filepath + var_import_df_filename)

    perf_metric_df_filepath = f'/scratch/w97/mg5624/data/trend_analysis/performance_metrics/{drought_type}_drought/{season}/Random_Forest_Regression/'
    perf_metric_df_filename = f'{season}_{region}_{drought_type}_drought_{years}_trend_analysis_Random_Forest_Regression_std_dev.csv'
    if not os.path.exists(perf_metric_df_filepath):
        os.makedirs(perf_metric_df_filepath)
    perfomance_metrics.to_csv(perf_metric_df_filepath + perf_metric_df_filename)


PREDICTORS = {
    'runoff': [
        'ET', 
        'Precipitation', 
        'Soil_Moisture'
    ],
    'soil_moisture': [
        'ET',
        'Precipitation', 
        'Runoff'
    ],
}


NRM_REGIONS = [
    'Central_Slopes',
    'East_Coast',
    'Murray_Basin',
    'Monsoonal_North',
    'Rangelands',
    'Southern_Slopes',
    'S_SW_Flatlands',
    'Wet_Tropics',
]


SEASONS = ['DJF', 'MAM', 'JJA', 'SON']


def main():
    arguments = []
    for drought_type in [
        'runoff', 
        'soil_moisture',
        ]:
        for region in NRM_REGIONS:
            for season in SEASONS:
                arguments.append(
                    (
                        drought_type, 
                        '1981-2020', 
                        PREDICTORS[drought_type], 
                        region,
                        season,
                        True, #add_random argument
                    )
                )

    nb_cpus = 8
    pool = multiprocessing.Pool(processes=nb_cpus)
    pool.starmap(save_performance_metric_and_variable_importance_scores, arguments)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
