# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
##### ------------------------------------------------------------------- #####

def compute_ss(matrix, group_indices):
    """
    Compute the sum of squares for a given group.
    
    Args:
        matrix (np.array): The distance matrix.
        group_indices (list): Indices of the samples belonging to the group.
    
    Returns:
        float: The sum of squares for the given group.
    """
    group_centroid = np.mean(matrix[group_indices], axis=0)
    ss = np.sum((matrix[group_indices] - group_centroid)**2)
    return ss

def permute_groups(grouping):
    """
    Permute the group labels.
    
    Args:
        grouping (np.array): Array of group labels.
    
    Returns:
        np.array: Permuted group labels.
    """
    return np.random.permutation(grouping)

def perm_anova(distance_matrix, grouping, permutations=999):
    """
    Perform permutation-based ANOVA.
    
    Args:
        distance_matrix (np.array): The square distance matrix.
        grouping (np.array): Array of group labels.
        permutations (int, optional): Number of permutations. Defaults to 999.
    
    Returns:
        pd.Series: A series containing the pseudo-F statistic, p-value, and number of permutations.
    """
    n = len(grouping)
    unique_groups, group_counts = np.unique(grouping, return_counts=True)
    k = len(unique_groups)

    # Calculate within-group SS
    ss_within = 0
    for group in unique_groups:
        group_indices = np.where(grouping == group)[0]
        ss_within += compute_ss(distance_matrix, group_indices)

    # Calculate between-group SS
    total_centroid = np.mean(distance_matrix, axis=0)
    ss_between = 0
    for group, count in zip(unique_groups, group_counts):
        group_indices = np.where(grouping == group)[0]
        group_centroid = np.mean(distance_matrix[group_indices], axis=0)
        ss_between += count * np.sum((group_centroid - total_centroid)**2)

    # Calculate degrees of freedom
    df1 = k - 1  # df between
    df2 = n - k  # df within

    # Calculate pseudo-F statistic
    ms_within = ss_within / df2
    ms_between = ss_between / df1
    pseudo_f = ms_between / ms_within

    # Permutation test
    permuted_f = []
    for _ in range(permutations):
        permuted_grouping = permute_groups(grouping)
        permuted_ss_within = 0
        for group in unique_groups:
            group_indices = np.where(permuted_grouping == group)[0]
            permuted_ss_within += compute_ss(distance_matrix, group_indices)

        permuted_ms_within = permuted_ss_within / df2
        permuted_f.append(ms_between / permuted_ms_within)

    p_value = np.sum(np.array(permuted_f) >= pseudo_f) / permutations

    # Constructing the stats report string
    stats_report = f"pseudo-F({df1}, {df2}) = {pseudo_f:.3f}, p < {p_value:.3f}"

    # Constructing the series
    result = pd.Series([stats_report, pseudo_f, p_value, permutations], index=["Stats Report", "Pseudo-F", "p-value", "Permutations"])

    return result
def perform_perm_anova(data, group_labels, permutations=1000):
    """
    Perform permANOVA on a dataframe.
    
    Args:
        dataframe (np.array): Matrix of values- each column is a different variable.
        group_labels (list): list of group labels.
        permutations (int, optional): Number of permutations. Defaults to 1000.
    
    Returns:
        dict: A dictionary containing the pseudo-F statistic, p-value, and number of permutations.
    """

    # Calculate the distance matrix from the dataframe
    data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
    distance_matrix = pdist(data, metric='euclidean')
    square_distance_matrix = squareform(distance_matrix)
    
    # Perform permANOVA
    perm_anova_results = perm_anova(square_distance_matrix, group_labels, permutations=permutations)
    
    return perm_anova_results


    

if __name__ == '__main__':

    
    ### Third Example ###
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    def generate_dataset(effect_size, distribution='normal'):
        if distribution == 'normal':
            return np.concatenate([np.random.normal(loc=0, scale=2, size=(50, 4)),
                                   np.random.normal(loc=effect_size, scale=2, size=(50, 4))])
        elif distribution == 'uniform':
            return np.concatenate([np.random.uniform(low=0, high=1, size=(50, 4)),
                                   np.random.uniform(low=effect_size, high=effect_size + 1, size=(50, 4))])
        else:
            raise ValueError('Invalid distribution type. Supported types: "normal", "uniform", "exponential".')
        
    # Generate 20 synthetic datasets with varying degrees of separation
    effect_sizes = np.linspace(0, 2, 40)
    datasets = [generate_dataset(effect_size, distribution='normal') for effect_size in effect_sizes]
    
    # Assign group labels
    group_labels = np.array(['Group1'] * 50 + ['Group2'] * 50)
    
    # Run permANOVA on each dataset
    p_values = []
    for data, effect_size in zip(datasets, effect_sizes):
        distance_matrix = pdist(data, metric='euclidean')
        square_distance_matrix = squareform(distance_matrix)
        perm_anova_results = perm_anova(square_distance_matrix, group_labels, permutations=1000)
        p_values.append(perm_anova_results['p-value'])
        # print(f"Effect size {effect_size:.2f} permANOVA results:", perm_anova_results)
    
    # Plot the input difference (effect size) vs the p-value
    plt.plot(effect_sizes, p_values, marker='o')
    plt.xlabel('Effect size (input difference)')
    plt.ylabel('p-value')
    plt.title('Effect size vs p-value for permANOVA')
    plt.axhline(y=0.05, linestyle='--', color='red', label='p=0.05')
    plt.legend()
    plt.show()































