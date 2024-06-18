# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.api import add_constant
##### ------------------------------------------------------------------- #####

def tukeys_fences(data, k=1.5):
    """
    This function identifies and removes outliers from a dataset using Tukey's Fences method.
    
    Inputs:
    - data: A list or NumPy array of numerical values
    - k: A constant multiplier to determine the threshold for identifying outliers (default: 1.5)
    
    Returns:
    - data_without_outliers: A NumPy array containing the data points within the bounds
    - outliers: A NumPy array containing the detected outlier data points
    """
    # Calculate the first quartile (Q1) and the third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    # Convert data to a NumPy array for logical indexing
    data = np.array(data)

    # Identify the outliers and non-outliers using logical indexing
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    data_without_outliers = data[(data >= lower_bound) & (data <= upper_bound)]

    return data_without_outliers, outliers


def rout(data_x, data_y, n_iterations=5, q=1e-5):
    """
    This function applies the ROUT method to identify and remove outliers from a dataset with a strong underlying trend or structure.

    Inputs:
    - data_x: A list or NumPy array of independent variable values
    - data_y: A list or NumPy array of dependent variable values
    - n_iterations: Number of iterations to perform (default: 5)
    - q: Quantile for outlier detection (default: 1e-5)

    Returns:
    - inliers_x: A NumPy array containing the inlier data points of the independent variable
    - inliers_y: A NumPy array containing the inlier data points of the dependent variable
    - outliers_x: A NumPy array containing the detected outlier data points of the independent variable
    - outliers_y: A NumPy array containing the detected outlier data points of the dependent variable
    """
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    inliers_mask = np.ones(len(data_x), dtype=bool)

    for _ in range(n_iterations):
        # Fit a robust linear model to the data
        X = add_constant(data_x[inliers_mask])
        Y = data_y[inliers_mask]
        model = RLM(Y, X, M=sm.robust.norms.HuberT()).fit()

        # Calculate standardized residuals
        residuals = model.resid
        standardized_residuals = (residuals - np.median(residuals)) / sm.robust.scale.mad(residuals)

        # Identify and remove outliers
        outlier_mask = (standardized_residuals ** 2 > -2 * np.log(q))
        inliers_mask[inliers_mask] = ~outlier_mask

    inliers_x = data_x[inliers_mask]
    inliers_y = data_y[inliers_mask]
    outliers_x = data_x[~inliers_mask]
    outliers_y = data_y[~inliers_mask]

    return inliers_x, inliers_y, outliers_x, outliers_y



if __name__ == '__main__':
    # =============================================================================
    # Example
    # =============================================================================
    
    """
    In this example, we have two cases:
    
    "tukey_good_rout_bad": This dataset is designed to showcase a situation where Tukey's Fences performs well, and ROUT performs poorly. The dataset has a constant noise distribution (independent of the x-axis), which is suitable for Tukey's Fences but unfavorable for ROUT due to the higher noise level.
    
    "tukey_bad_rout_good": This dataset is designed to showcase a situation where Tukey's Fences performs poorly, and ROUT performs well. The dataset has a noise distribution that increases linearly with the x-axis, making it suitable for ROUT but not for Tukey's Fences due to the non-constant noise distribution.
    
    "tukey_good_rout_moderate": In this case, Tukey's Fences performs well, and ROUT performs moderately. The dataset has a constant noise distribution (independent of the x-axis), similar to the "tukey_good_rout_bad" case but with a lower noise level. The lower noise level allows ROUT to perform moderately well, but Tukey's Fences is still better suited to handle constant noise.

    "tukey_moderate_rout_good": In this case, Tukey's Fences performs moderately, and ROUT performs well. The dataset has a noise distribution that increases quadratically with the x-axis. This non-constant noise distribution is challenging for Tukey's Fences but is still manageable to some extent, while ROUT can handle the increasing noise better due to its robust regression approach.
    
    These plots will provide a clear illustration of the strengths and weaknesses of each method, based on the characteristics of the dataset.
    
    """
    
    # Helper function to generate synthetic datasets
    def generate_data(case):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 3 + np.random.normal(0, 2, len(x))
    
        if case == "tukey_good_rout_bad":
            y[[15, 60, 80]] += 20
            y += 5 * np.random.normal(0, 1, len(x))
        elif case == "tukey_bad_rout_good":
            y[[15, 60, 80]] += 20
            y += 0.3 * x * np.random.normal(0, 1, len(x))
        elif case == "tukey_good_rout_moderate":
            y[[15, 60, 80]] += 15
            y += 2 * np.random.normal(0, 1, len(x))
        else:  # case == "tukey_moderate_rout_good"
            y[[15, 60, 80]] += 20
            y += 0.2 * x ** 2 * np.random.normal(0, 1, len(x))
    
        return x, y
    
    cases = ["tukey_good_rout_bad", "tukey_bad_rout_good", "tukey_good_rout_moderate", "tukey_moderate_rout_good"]
    
    # Modified plotting function for subplots
    def plot_comparison(ax, x, y, inliers_x, inliers_y, outliers_x, outliers_y, case, method):
        ax.scatter(inliers_x, inliers_y, label="Inliers", color="blue")
        ax.scatter(outliers_x, outliers_y, label="Outliers", color="red", marker="x")
        ax.plot(x, y, "o", alpha=0.3, label="Original Data", color="gray")
        ax.set_title(f"{method} Method - {case.capitalize()} Performance")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid()
    
    # Create a single figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    fig.tight_layout(pad=5.0)

    for i, case in enumerate(cases):
        x, y = generate_data(case)
        data_without_outliers, outliers = tukeys_fences(y)
        inliers_x_tukey = x[(y >= min(data_without_outliers)) & (y <= max(data_without_outliers))]
        inliers_y_tukey = y[(y >= min(data_without_outliers)) & (y <= max(data_without_outliers))]
        outliers_x_tukey = x[(y < min(data_without_outliers)) | (y > max(data_without_outliers))]
        outliers_y_tukey = y[(y < min(data_without_outliers)) | (y > max(data_without_outliers))]
        plot_comparison(axes[i, 0], x, y, inliers_x_tukey, inliers_y_tukey, outliers_x_tukey, outliers_y_tukey, case, "Tukey's Fences")
        
        inliers_x_rout, inliers_y_rout, outliers_x_rout, outliers_y_rout = rout(x, y)
        plot_comparison(axes[i, 1], x, y, inliers_x_rout, inliers_y_rout, outliers_x_rout, outliers_y_rout, case, "ROUT")
    
    plt.show()




















