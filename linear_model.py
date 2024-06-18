# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as skmetrics
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
##### ------------------------------------------------------------------- #####

def remove_corr_features(df: pd.DataFrame, num_cols: list, r_threshold: float = 0.95) -> tuple:
    """
    This function takes a pandas DataFrame, a list of numeric column names, and a correlation threshold and returns a tuple
    containing the set of selected features and the set of features to be dropped due to high correlation.

    Parameters:
    - df (pandas DataFrame): input DataFrame
    - num_cols (list): list of column names for numeric features
    - r_threshold (float, optional): correlation threshold, default value is 0.95

    Returns:
    - tuple: contains two lists of column names - one list for selected features, another list for dropped features
    """

    # calculate correlation matrix
    corr_matrix = np.corrcoef(df[num_cols].T)
    corr_matrix = pd.DataFrame(corr_matrix, index=num_cols, columns=num_cols)
    
    # select upper triangle of correlation matrix
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # find highly correlated feature pairs
    correlated_pairs = corr_matrix[corr_matrix >= r_threshold].stack().index.tolist()
    
    if len(correlated_pairs) == 0:
        return list(num_cols), []
        
    # select features from highly correlated pairs
    corr_feature_df = pd.DataFrame(correlated_pairs)
    high_corr_features = set(np.unique(corr_feature_df[corr_feature_df.columns].values))
    for feature1, feature2 in correlated_pairs:
        corr_feature_df = corr_feature_df[corr_feature_df.iloc[:,0] != feature2]
    
    # get selected and dropped features
    selected_high_corr_features = set(corr_feature_df.iloc[:,0].values)
    drop_features = high_corr_features.difference(selected_high_corr_features)
    selected_features = set(corr_matrix.index).difference(drop_features)

    # return selected and dropped feature sets
    return list(selected_features), list(drop_features)


def run_pca(X, n_components, show_plot=True, threshold=.9):
    """
    Perform PCA on data matrix X and plot the cumulative sum of explained variance ratios.

    Parameters:
        X (ndarray): Data matrix of shape (n_samples, n_features).
        n_components (int): Number of principal components to use.

    Returns:
        A tuple containing:
        - pcs (ndarray): Principal components of shape (n_samples, n_components).
        - exp_var_ratios (ndarray): Explained variance ratios of shape (n_components,).
        - selected_compononets (int): Number of components selected based on threshold.

    """
    # create a PCA object with n_components
    pca = PCA(n_components=n_components)

    # fit the PCA model to the data
    pca.fit(X)

    # get the principal components
    pcs = pca.transform(X)

    # get the explained variance ratios
    exp_var_ratios = pca.explained_variance_ratio_

    # select number of features below threshold
    cum_sum = np.cumsum(exp_var_ratios)
    selected_compononets = np.argmax(cum_sum >= threshold) + 1

    if show_plot:
        # plot the cumulative sum of explained variance ratios
        plt.plot(np.cumsum(exp_var_ratios))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Sum of Explained Variance')
        plt.show()

    return pcs, exp_var_ratios, selected_compononets


def find_best_covariance_type(X, n_clusters=2, n_init=10, max_iter=300):
    """
    Finds the best covariance type for a Gaussian mixture model based on the silhouette score.
    
    Parameters:
        X (numpy.ndarray): Input data with shape (n_samples, n_features).
        n_clusters (int): The number of clusters to test. Default is 2.
        n_init (int): The number of times the model will be run with different
            initializations. Default is 10.
        max_iter (int): The maximum number of iterations for the model. Default is 300.
        
    Returns:
        The optimal covariance type.
    """
    # Define the covariance types to test
    covariance_types = ['full', 'tied', 'diag', 'spherical']
 
    best_score = -1
    best_covariance_type = None

    for covariance_type in covariance_types:
        # Create a Gaussian mixture model with the current covariance type
        gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                              n_init=n_init, max_iter=max_iter, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        silhouette_avg = silhouette_score(X, labels)

        # Update the best score and parameters if the current score is better
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_covariance_type = covariance_type

    return best_covariance_type


def train_gmm(X, n_clusters=2, n_init=10, max_iter=300):
    """
    Trains a Gaussian mixture model with the best covariance type found by the
    find_best_covariance_type function and returns the predicted labels.
    
    Parameters:
        X (numpy.ndarray): Input data with shape (n_samples, n_features).
        n_clusters (int): The number of clusters to use. Default is 2.
        n_init (int): The number of times the model will be run with different
            initializations. Default is 10.
        max_iter (int): The maximum number of iterations for the model. Default is 300.
        
    Returns:
        The predicted cluster labels for the input data.
    """
    # Find the best covariance type
    covariance_type = find_best_covariance_type(X, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    
    # Create a Gaussian mixture model with the best covariance type
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                          n_init=n_init, max_iter=max_iter, random_state=42)
    
    # Train the model on the input data and predict cluster labels
    gmm.fit(X)
    labels = gmm.predict(X)
    
    return labels


def find_best_model(x, y, n_splits=5):
    """
    Finds the best hyperparameters for a Gaussian Naive Bayes model using grid search
    with cross-validation and returns the best parameters.

    Parameters:
    - x: a 2D array or DataFrame of shape (n_samples, n_features) representing the input data
    - y: a 1D array or Series of shape (n_samples,) representing the target labels
    - n_splits: an integer representing the number of folds in cross-validation (default: 5)

    Returns:
    - A dictionary representing the best hyperparameters found by grid search
    """

    # Define the hyperparameter grid to search over
    param_grid = {
        'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    # Create an instance of the Gaussian Naive Bayes model
    model = GaussianNB()

    # Create a cross-validation iterator
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=11)

    # Create a GridSearchCV object with the model, hyperparameter grid, and cross-validation iterator
    grid = GridSearchCV(model, param_grid, cv=kfold, scoring='balanced_accuracy')

    # Fit the GridSearchCV object on the input data and target labels
    grid.fit(x, y)

    # Return the best hyperparameters found by grid search
    return grid.best_params_


def train_best_model(x, y, n_splits=5):
    """
    Trains a Gaussian Naive Bayes model on 80% of the shuffled data based on the best hyperparameters
    found by the find_best_model function.

    Args:
    - x: a 2D array or DataFrame of shape (n_samples, n_features) representing the input data
    - y: a 1D array or Series of shape (n_samples,) representing the target labels
    - n_splits: an integer representing the number of folds in cross-validation (default: 5)

    Returns:
    - model: A trained Gaussian Naive Bayes model
    - best_params: list of best parameters used to train the model
    """

    # Use the find_best_model function to find the best hyperparameters
    best_params = find_best_model(x, y, n_splits)

    # Create an instance of the Gaussian Naive Bayes model with the best hyperparameters
    model = GaussianNB(var_smoothing=best_params['var_smoothing'])

    # Split the input data and target labels into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=11)

    # Fit the model on the training data
    model.fit(x_train, y_train)

    # Return the trained model
    return model , best_params


def assign_cluster_rmp(df, group_column='projection'):
    """
    Assign clusters to a dataframe based on the average resting membrane potential (RMP) for each group.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe to assign clusters to.
    group_column : str, optional
        The name of the column that specifies the group to cluster by. Default is 'projection'.
    
    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with an additional 'cluster' column that specifies the assigned cluster.
    """
    df = df.copy()
    avg = df.groupby(by=[group_column])['resting_membrane_potential'].mean()
    categories = df[group_column].unique()
    df['cluster'] = ''
    if avg[categories[0]] > avg[categories[1]]:
        df.loc[df[group_column] == categories[0], 'cluster'] = 'cluster+'
        df.loc[df[group_column] == categories[1], 'cluster'] = 'cluster-'
    elif avg[categories[1]] > avg[categories[0]]:
        df.loc[df[group_column] == categories[0], 'cluster'] = 'cluster-'
        df.loc[df[group_column] == categories[1], 'cluster'] = 'cluster+'
    return df


def test_model(df, condition, cluster_method):
    """
    Test a machine learning model on a given dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset to test the model on.
    condition : str
        A condition to filter the dataset by.
    cluster_method : list of dict
        A list of dictionaries that specify how to cluster the data.
    
    Returns
    -------
    list of tuple: A list of tuples, where each tuple contains the balanced accuracy, F1 score, and best hyperparameters
            for a specific clustering method.

    """
    
    # normalize and split dataset in test and train based on condition
    num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    df_test = df[(df['treatment'].isin([condition])) & (~df['projection'].isin(['nonspecific']))].reset_index(drop=True)
    df_test = assign_cluster_rmp(df_test, group_column='projection')
    df_control = df[(df['treatment'].isin(['control'])) & (df['projection'].isin(['nonspecific']))].reset_index(drop=True)
    df = df[(df['treatment'].isin([condition])) & (df['projection'].isin(['nonspecific']))].reset_index(drop=True)
    
    # convert labels 
    convert = LabelEncoder().fit(df_test['cluster'])
    scores = []
    for c in tqdm(cluster_method, total=len(cluster_method)):
        
        print(c)
        therehold = c['threshold']
        
        if c['split'] == 'mode_split':
            temp = df.copy()
            feature = c['features'][0]
            split_threshold = df_control[feature].mode()[0]
            std = df_control[feature].std()
            temp['manual_cluster'] = ''
            bool_array = temp[feature] >= (split_threshold + (therehold*std))
            temp.loc[bool_array.values.flatten(), 'manual_cluster'] = 'cluster_1'
            bool_array = temp[feature] < (split_threshold - (therehold*std))
            temp.loc[bool_array.values.flatten(), 'manual_cluster'] = 'cluster_2'
            temp = temp[temp['manual_cluster'] != '']
            temp = assign_cluster_rmp(temp, group_column='manual_cluster')

            # get train and test array
            nan_mask = ~temp[feature].isna()
            X_train = temp[feature][nan_mask].values.reshape(-1, 1)
            y_train = convert.transform(temp['cluster'][nan_mask])
            nan_mask = ~df_test[feature].isna()
            X_test = df_test[feature][nan_mask].values.reshape(-1, 1)
            y_test = convert.transform(df_test['cluster'][nan_mask])
            
        elif c['split'] == 'gmm_split':
            features = c['features']
            
            if c['feature_type'] == 'original_feature':
                temp = df.copy()
                temp = temp[features].dropna(axis=0)
                labels = train_gmm(temp[features].values, n_clusters=2, n_init=10, max_iter=300)
                temp['gmm_cluster'] = labels
                temp = assign_cluster_rmp(temp, group_column='gmm_cluster')
                
                # get train and test array
                X_train = temp[features].values.reshape(-1, len(features))
                y_train =convert.transform(temp['cluster'])
                nan_mask = ~df_test[features].isna().any(axis=1)
                X_test = df_test[features][nan_mask].values.reshape(-1, len(features))
                y_test = convert.transform(df_test['cluster'][nan_mask])
                
            elif c['feature_type'] == 'pca':

                # drop nans from rows and remove highly correlated features before performing PCA
                temp = df.dropna(axis=0)
                selected_features, drop_features = remove_corr_features(temp, num_cols)
                
                # select PCA features
                pcs, exp_var_ratios, selected_compononets = run_pca(temp[selected_features], n_components=8, show_plot=False)
                selected_pcs = pcs[:, :selected_compononets]
                labels = train_gmm(selected_pcs, n_clusters=2, n_init=10, max_iter=300)
                temp['gmm_cluster'] = labels
                temp = assign_cluster_rmp(temp, group_column='gmm_cluster')
                
                # get train array
                y_train = convert.transform(temp['cluster'])
                X_train = selected_pcs.reshape(-1, selected_compononets)
                
                # drop nans from rows and select PCA features
                temp = df_test.dropna(axis=0)
                pcs, exp_var_ratios, selected_compononets = run_pca(temp[selected_features], n_components=10, show_plot=False)
                selected_pcs = pcs[:, :selected_compononets]

                # get test array
                X_test = selected_pcs.reshape(-1, selected_compononets)
                y_test = convert.transform(temp['cluster'])
                
        # train and test model
        model, best_params = train_best_model(X_train, y_train, n_splits=3)
        y_pred = model.predict(X_test)
        # y_prob = model.predict_proba(X_test)
        balanced_accuracy = skmetrics.balanced_accuracy_score(y_test, y_pred)
        f1 = skmetrics.f1_score(y_test, y_pred)
        # auc = skmetrics.roc_auc_score(y_test, y_prob[:,-1])
        scores.append({'balanced_accuracy':balanced_accuracy, 'f1_score':f1,})
        #'var_smoothing':best_params['var_smoothing']
    return scores


# Function to save the styled dataframe as an HTML file
def save_styled_df_to_html(styled_df, filepath):
    # Render the HTML with styles
    html = styled_df.to_html()

    # Add necessary CSS styles for the table
    table_style = """
    <style>
    table {
        border-collapse: collapse;
        font-family: 'Times New Roman', Times, serif;
        width: 100%;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
        padding: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }
    td {
        padding: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    </style>
    """

    # Combine the table style and HTML content
    full_html = f"<!DOCTYPE html><html><head>{table_style}</head><body>{html}</body></html>"

    # Save the full HTML to a file
    with open(filepath, "w") as f:
        f.write(full_html)

def style_df(df, path='styled_dataframe.html'):
    """
    Styles a pandas DataFrame and saves it as an HTML file.
    
    Args:
        df (pandas.DataFrame): The DataFrame to style.
        path (str, optional): The file path to save the styled DataFrame as an HTML file. 
                              Default is 'styled_dataframe.html'.
    """
    
    # Clean column name
    new_column_names = [col.replace('_', ' ').title() for col in df.columns]
    df = df.rename(columns=dict(zip(df.columns, new_column_names)))
    
    # clean Features
    df['Features'] = df['Features'].apply(lambda x: ', '.join(x))
        
    # apply colormap
    numerical_columns = df.select_dtypes(include=['number']).columns
    custom_blues = sns.cubehelix_palette(start=2.4, rot=0.2, light=0.95, dark=0.5,as_cmap=True)
    styled_df = df.style.background_gradient(cmap=custom_blues, subset=numerical_columns)
    
    # Save styled DataFrame to HTML file
    save_styled_df_to_html(styled_df, path)

if __name__ == '__main__':
    
    # get properties
    properties = pd.read_csv(os.path.join('clean_data', 'all_properties.csv'))
    io_columns = ['fr_at_20_percent_input', 'fr_at_40_percent_input', 'fr_at_60_percent_input', 'fr_at_max_input', 
                  'i_amp_at_half_max_fr', 'input_resistance', 'io_slope', 'max_firing_rate', 'resting_membrane_potential', 'rheobase']
    wave_columns = ['ap_peak', 'ahp', 'peak_to_trough', 'threshold', 'rise_time', 'half_width']

    # define cluster methods
    cluster_method = [
        {'split':'mode_split', 'feature_type':'original_feature', 'threshold':0, 'features':['resting_membrane_potential']},
        {'split':'mode_split', 'feature_type':'original_feature', 'threshold':1, 'features':['resting_membrane_potential']},
        {'split':'mode_split', 'feature_type':'original_feature', 'threshold':0, 'features':['rheobase']},
        {'split':'mode_split', 'feature_type':'original_feature', 'threshold':1, 'features':['rheobase']},
        
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features':['resting_membrane_potential']},
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features':['resting_membrane_potential', 'rheobase']},
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features':['resting_membrane_potential', 'fr_at_20_percent_input']},
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features':['resting_membrane_potential', 'input_resistance']},
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features':['resting_membrane_potential', 'rheobase', 'fr_at_20_percent_input','input_resistance']},
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features': io_columns},
        {'split':'gmm_split', 'feature_type':'original_feature', 'threshold':'n/a', 'features':['resting_membrane_potential'] + wave_columns},
        
        {'split':'gmm_split', 'feature_type':'pca', 'threshold':.8, 'features': ['all']},
        {'split':'gmm_split', 'feature_type':'pca', 'threshold':.9, 'features': ['all']},
        ]
    
    # get test scores
    scores = test_model(properties, 'cus', cluster_method)
    df_cus = pd.concat((pd.DataFrame(scores), pd.DataFrame(cluster_method)),axis=1)
    style_df(df_cus, 'cus_model.html')
    
    scores = test_model(properties, 'enriched-environment', cluster_method)
    df_ee = pd.concat((pd.DataFrame(scores), pd.DataFrame(cluster_method)),axis=1)
    style_df(df_ee, 'enriched-environment_model.html')
    
    
    











