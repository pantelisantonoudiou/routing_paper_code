# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from figure4_cus_behavior import fill_na_with_median, evaluate_gmm, plot_pca_gmm
##### ------------------------------------------------------------------- #####

if __name__ == '__main__':

    # load data
    df = pd.read_csv(os.path.join('clean_data', 'ee_behavior.csv'), keep_default_na=True)

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
    npcs = 5
    data_pca = pcs[:,0:npcs]
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
    
    
    clusters = ['Cluster ' + str(i+1) for i in range(best_n_clusters)]
    df = pd.DataFrame(percentage_per_group, columns=clusters, index=groups)
    df = df.reindex(['control', 'bnst+nac-', 'enriched-environment', 'bnst-nac+',])
    ax = df.plot(kind='bar', stacked=True)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Groups')
    plt.xticks(rotation=0)
    plt.legend(title='Clusters')
    plt.tight_layout()
    
    # 3) Plot decision boundary scatter plots
    plot_pca_gmm(pcs, npcs, best_n_clusters, best_covar_type, actual_labels, groups, cols=4)

  
  
  