# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from itertools import combinations
import seaborn as sns
##### ------------------------------------------------------------------- #####


def fill_na_with_median(df, group_variable):
    """
    Converts empty strings to NaNs and then fills missing values 
    in each numerical variable with the median of that variable in each group.

    Parameters:
    df (DataFrame): The DataFrame to modify.
    group_variable (str): The name of the group identifier column.

    Returns:
    df (DataFrame): The modified DataFrame with missing values filled.
    """
    # # Convert empty strings to NaNs
    # df = df.replace('', pd.NaT)

    # Get list of numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Fill missing values with the median of each group
    for column in numerical_columns:
        df[column] = df.groupby(group_variable)[column].transform(lambda x: x.fillna(x.median()))
    
    return df

def evaluate_gmm(data_pca, n_clusters_range, covar_types=['full', 'tied', 'diag', 'spherical']):

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['n_clusters', 'covariance_type', 'aic', 'bic', 'silhouette_score'])

    # Fit GMMs with different parameters
    for n in n_clusters_range:
        for covar_type in covar_types:
            # Fit the GMM
            gmm = GaussianMixture(n_components=n, covariance_type=covar_type, random_state=42)
            gmm.fit(data_pca)

            # Compute the AIC and BIC
            aic = gmm.aic(data_pca)
            bic = gmm.bic(data_pca)

            # Predict the labels and compute silhouette_score
            labels = gmm.predict(data_pca)
            silhouette = silhouette_score(data_pca, labels)

            # Store the results
            df = pd.DataFrame([{'n_clusters': n, 
                                'covariance_type': covar_type, 
                                'aic': aic, 
                                'bic': bic, 
                                'silhouette_score': silhouette}])
            results = pd.concat([results, df], ignore_index=True)
    return results

def plot_pca_gmm(pcs, num_pcs, best_n_clusters, best_covar_type, actual_labels, groups, cols=3):
    """
    Visualize data and decision boundaries for a Gaussian Mixture Model (GMM) fitted to Principal Component Analysis (PCA) projected data.
    
    Parameters:
    pcs: array-like
        The data, already projected onto principal components.
    num_pcs: int
        The number of principal components to use.
    best_n_clusters: int
        The optimal number of clusters for the GMM.
    best_covar_type: str
        The type of covariance parameters to use.
    actual_labels: array-like
        The actual labels of the data.
    groups: list
        The unique group labels.
    cols: int, optional
        The number of columns for subplot arrangement. Default is 3.
    
    Returns:
    None
    """
    
    # Adjust PCs based on num_pcs
    data_pca = pcs[:, :num_pcs]
    gmm = GaussianMixture(n_components=best_n_clusters, covariance_type=best_covar_type, random_state=42)
    gmm.fit(data_pca)

    # All combinations of the PCs
    pairs = list(combinations(range(num_pcs), 2))  

    # Generate a distinct color for each group
    distinct_colors = sns.color_palette("hsv", len(groups))
    colors = {group: distinct_colors[i] for i, group in enumerate(groups)}
    
    rows = int(np.ceil(len(pairs)/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4*rows))

    for ax, pair in zip(axs.flatten(), pairs):
        x, y = pair

        # Create a meshgrid for the decision boundaries
        x_min, x_max = data_pca[:, x].min() - 1, data_pca[:, x].max() + 1
        y_min, y_max = data_pca[:, y].min() - 1, data_pca[:, y].max() + 1
        hx = (x_max - x_min)/100.
        hy = (y_max - y_min)/100.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

        # Fill other dimensions with their mean values
        grid = np.full((xx.ravel().shape[0], num_pcs), data_pca.mean(axis=0))
        grid[:, x] = xx.ravel()
        grid[:, y] = yy.ravel()
        
        # Predict the labels for each point in the meshgrid
        Z = gmm.predict(grid).reshape(xx.shape)

        # Make scatter plot with color-coded groups
        for group in groups:
            indices = np.where(actual_labels == group)[0]
            ax.scatter(data_pca[indices, x], data_pca[indices, y], color=colors[group], label=group)

        # Plot the decision boundary
        ax.contourf(xx, yy, Z, levels=best_n_clusters-1, colors=list(colors.values()), alpha=.1)

        ax.set_title(f'PC{x+1} vs PC{y+1}')
        ax.set_xlabel(f'PC{x+1}')
        ax.set_ylabel(f'PC{y+1}')

        # Create a legend for each subplot
        ax.legend(title='Group')

    plt.tight_layout()
if __name__ == '__main__':

    # load data
    df = pd.read_csv(os.path.join('clean_data', 'cus_behavior.csv'), keep_default_na=True)

    # remove nans where more than half of the parameters are missing
    num_cols = df.select_dtypes(include=[np.number]).columns
    mask = df[num_cols].isna().sum(axis=1) < len(num_cols) / 2
    df = df.loc[mask]
    
    # fill missing values
    df = fill_na_with_median(df, group_variable='treatment')
    actual_labels = df['treatment']
    
    # normalize
    df[num_cols] =  (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    pca = PCA(n_components=len(num_cols))
    pca.fit(df[num_cols].values)
    
    # get pcs
    pcs = pca.transform(df[num_cols].values)
    
    # 1) Plot explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    plt.figure()
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    cutoff = 0.9
    plt.axhline(y=cutoff, color='orange', linestyle='--')
    plt.annotate(f'{cutoff}', xy=(0, cutoff), xytext=(3, 0),
                 color='r', ha='left', va='center', fontweight='bold')
    plt.tight_layout()
    
    # choose pcs to train gmm
    npcs=3
    data_pca = pcs[:,:npcs]
    groups = actual_labels.unique()
    
    # get metrics for gmm
    n_clusters_range = range(2, 5)
    results = evaluate_gmm(data_pca, n_clusters_range)
    
    # Get the best number of clusters and covariance matrix type based on Silhouette Score
    best_silhouette = results.loc[results['silhouette_score'].idxmax()]
    best_n_clusters = int(best_silhouette['n_clusters'])
    best_covar_type = best_silhouette['covariance_type']
    
    # Train the GMM using the best parameters
    gmm = GaussianMixture(n_components=best_n_clusters, covariance_type=best_covar_type, random_state=42)
    gmm.fit(data_pca)
    
    # Get the predicted labels from the GMM
    predicted_labels = gmm.predict(data_pca)
    
    # 2) Plot percentage of each group per cluster
    percentage_per_group = []
    for group in groups:
        group_indices = np.where(actual_labels == group)[0]
        group_predicted_labels = predicted_labels[group_indices]
        percentage_per_cluster = []
        for cluster in range(best_n_clusters):
            cluster_count = np.sum(group_predicted_labels == cluster)
            percentage = cluster_count / len(group_predicted_labels)
            percentage_per_cluster.append(percentage*100)
        percentage_per_group.append(percentage_per_cluster)
    
    clusters = ['Cluster 1', 'Cluster 2']
    df = pd.DataFrame(percentage_per_group, columns=clusters, index=groups)
    df = df.reindex(['control', 'bnst-nac+', 'cus', 'bnst+nac-',])
    ax = df.plot(kind='bar', stacked=True)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Groups')
    plt.xticks(rotation=0)
    plt.legend(title='Clusters')
    plt.tight_layout()
    
    # 3) Plot decision boundary scatter plots
    plot_pca_gmm(pcs, npcs, best_n_clusters, best_covar_type, actual_labels, groups, cols=4)
  