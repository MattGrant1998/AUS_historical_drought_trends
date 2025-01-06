import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

plotdir = '/g/data/w97/mg5624/plots/RF_project/model_analytics/'
datadir = '/g/data/w97/mg5624/RF_project/'

training_data = pd.read_csv(datadir + 'training_data/training_data.csv')
training_data.dropna(axis=0, inplace=True)

impact_predictors = [
        'Nino_index',
        'SOI_index',
        'Acc_12-Month_Precipitation', 
        'Mean_12-Month_Runoff',
        'IOD_index', 
        'Cos_month', 
        'Sin_month',
        'Mean_12-Month_AWRA_SM',
]

target = 'Drought'
y = training_data['Drought']


def calculate_performance_metrics(y_test, y_pred):
    """
    Calculates performance metrics of the Random Forest (RF) model when predicting drought.
    The function calculates the following performance metrics:
    - Accuracy: The ratio of correctly predicted instances to the total instances.
    - Precision: The ratio of correctly predicted positive observations to the total predicted positives.
    - Recall: The ratio of correctly predicted positive observations to all observations in the actual class.
    - F1-Score: The weighted average of precision and recall.
    - Balanced Accuracy: The average of recall obtained on each class.
    - False Alarm: The proportion of negative instances that were incorrectly classified as positive.

    Args:
        y_test (array-like): Data held back from data split for testing.
        y_pred (array-like): Predictions made by the RF Classifier model.
    Returns:
        performance_df (pd.DataFrame): A DataFrame containing the performance metrics including accuracy, precision, recall, F1-score, balanced accuracy, and false alarm rate.
    """
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all')
    false_alarm = confusion_matrix[0, 1]
    
    # Save results in DataFrame
    performance_data = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'Balance Accuracy': [balanced_accuracy],
        'False Alarm': [false_alarm]
    }

    performance_df = pd.DataFrame(performance_data)
    
    return performance_df


def performance_and_variable_importance_from_n_iterated_RF_model_seeds(X, y, test_size, var_import_type='MDI', n_iterations=100):
    """
    Trains the Random Forest with different seeds to assess the stability and generalizability of the model.
    
    Args:
        X (pd.DataFrame): Predictor variables data.
        y (pd.Series): Target variable data.
        test_size (float): Proportion of data to be held back for testing.
        var_import_type (str): The type of variable importance to compute ('MDI', 'permutation_all', or 'permutation_unseen').
        n_iterations (int): Number of iterations of the model (default=100).
    
    Returns:
        performance_df (pd.DataFrame): DataFrame containing performance metrics (accuracy, precision, recall, F1-score, balanced accuracy, and false alarm rate) for each iteration.
        variable_importance_df (pd.DataFrame): DataFrame containing variable importance scores for each iteration.
    """
    seeds = np.arange(n_iterations)

    # Initialize arrays to store variable importance
    variable_importance = np.zeros((len(X.columns), n_iterations))

    # Initialize arrays to store performance metrics
    accuracy_scores = np.zeros(n_iterations)
    precision_scores = np.zeros(n_iterations)
    recall_scores = np.zeros(n_iterations)
    f1_scores = np.zeros(n_iterations)
    balanced_accuracy_scores = np.zeros(n_iterations)
    false_alarm_scores = np.zeros(n_iterations)

    # Train the model and calculate performance metrics for each iteration
    for i, seed in enumerate(seeds):
        # Split the data into training and testing sets for each iteration
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
        # Train the Random Forest model with a different seed
        clf = RandomForestClassifier(n_estimators=500, random_state=seed)
        clf.fit(X_train, y_train)

        # Calculate variable importance
        if var_import_type == 'MDI':
            variable_importance[:, i] = clf.feature_importances_
        elif var_import_type == 'permutation_all':
            perm_variable_importance_dict = permutation_importance(clf, X, y, n_repeats=50, random_state=42, n_jobs=-1)
            variable_importance[:, i] = perm_variable_importance_dict['importances_mean']
        elif var_import_type == 'permutation_unseen':
            perm_variable_importance_dict = permutation_importance(clf, X_test, y_test, n_repeats=50, random_state=42, n_jobs=-1)
            variable_importance[:, i] = perm_variable_importance_dict['importances_mean']

        # Predict on test data
        y_pred = clf.predict(X_test)
    
        # Calculate performance metrics
        accuracy_scores[i] = accuracy_score(y_test, y_pred)
        precision_scores[i] = precision_score(y_test, y_pred)
        recall_scores[i] = recall_score(y_test, y_pred)
        f1_scores[i] = f1_score(y_test, y_pred)
        balanced_accuracy_scores[i] = balanced_accuracy_score(y_test, y_pred)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, normalize='all')

        false_alarm_scores[i] = confusion_matrix[0, 1]
        

    # Create a DataFrame to store the performance metrics for each iteration
    performance_df = pd.DataFrame({
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-score': f1_scores,
        'Balanced Accuracy': balanced_accuracy_scores,
        'False Alarm': false_alarm_scores
    })

    variable_importance_df = pd.DataFrame(variable_importance.T, columns=X.columns)
    
    return performance_df, variable_importance_df


def find_mean_performance_metrics_and_var_importance(performance_df, variable_importance_df):
    """
    Calculate the mean of each performance metric and variable importance.

    Args:
        performance_df (pd.DataFrame): DataFrame containing performance metrics from each iteration.
        variable_importance_df (pd.DataFrame): DataFrame containing variable importance scores from each iteration.

    Returns:
        mean_performance_df (pd.DataFrame): DataFrame with the mean of each performance metric.
        mean_variable_importance_df (pd.DataFrame): DataFrame with the mean importance score of each variable.
    """
    mean_performance_df = performance_df.mean(axis=0).to_frame().T
    mean_variable_importance_df = variable_importance_df.mean(axis=0).to_frame().T
    
    return mean_performance_df, mean_variable_importance_df


def combine_all_iterative_functions(data, predictors, target, test_size, var_import_type='MDI', n_iterations=100):
    """
    Trains n_iterations RF models with different random states and calculates mean performance metrics and
    variable importance.

    Args:
        data (pd.DataFrame): Training data.
        predictors (list): List of all the predictors for the model.
        target (str): Target variable for the model.
        test_size (float): Proportion of training data to hold back for testing.
        var_import_type (str): The type of variable importance to compute ('MDI', 'permutation_all', or 'permutation_unseen').
        n_iterations (int): Number of RF models to train (default=100).

    Returns:
        performance_df (pd.DataFrame): DataFrame containing performance metrics for each iteration.
        mean_performance_df (pd.DataFrame): DataFrame containing the mean of each performance metric.
        variable_importance_df (pd.DataFrame): DataFrame containing variable importance scores for each iteration.
        mean_variable_importance_df (pd.DataFrame): DataFrame containing the mean importance score of each variable.
    """
    X = data[predictors]
    y = data[target]
    performance_df, variable_importance_df = performance_and_variable_importance_from_n_iterated_RF_model_seeds(X, y, test_size, var_import_type, n_iterations)
    mean_performance_df, mean_variable_importance_df = find_mean_performance_metrics_and_var_importance(performance_df, variable_importance_df)

    return performance_df, mean_performance_df, variable_importance_df, mean_variable_importance_df


def main():
    combine_all_iterative_functions(training_data, impact_predictors, target, 0.3)


if __name__ == "__main__":
    main()
    