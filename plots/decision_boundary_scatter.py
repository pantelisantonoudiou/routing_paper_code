# -*- coding: utf-8 -*-


##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import sklearn.metrics as skmetrics
from sklearn.impute import KNNImputer
from linear_model import train_best_model
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
##### ------------------------------------------------------------------- #####


def scatter_contour(properties, mapping={1:'#EFA0BA', 0:'#FBD3A5'}, bounds=5):
    """
    Train a model, plot the decision boundary, and calculate the AUC for Resting Membrane Potential.

    Parameters:
        properties (DataFrame): A DataFrame containing cell properties.
        mapping (dict, optional): A dictionary mapping group values to colors. Defaults to {1:'#EFA0BA', 0:'#FBD3A5'}.
        bounds (int, optional): The boundary value to expand the plot range. Defaults to 5.

    Returns:
        float: The calculated AUC (Area Under the Curve).

    """
    # Train and test the model using the provided properties
    train = properties[properties['projection'].isin(['nonspecific'])]
    X = train[['resting_membrane_potential', 'rheobase']].values
    y = train['group']
    model, best_params = train_best_model(X[:,0].reshape(-1, 1), y, n_splits=3)
    
    # Prepare the testing data and create a grid of points for plotting the decision boundary
    test = properties[~properties['projection'].isin(['nonspecific'])]
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X = imputer.fit_transform(test[['resting_membrane_potential', 'rheobase']])
    y = test['group']
    x_min, x_max = X[:, 0].min() - bounds, X[:, 0].max() + bounds
    y_min, y_max = X[:, 1].min() - bounds, X[:, 1].max() + bounds
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict using the model and plot the decision boundary
    Z = model.predict(xx.ravel().reshape(-1, 1))
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.1, cmap='PRGn')
    plt.scatter(X[:, 0], X[:, 1], c=y.map(mapping), edgecolors='k', s=50)
    plt.xlabel('Resting Membrane Potential')
    plt.ylabel('Rheobase')
    plt.title('GaussianNB decision boundary for Resting Membrane Potential')
    
    # Calculate the AUC
    X_train = train[['resting_membrane_potential']].values
    y_train = train['group']
    X_test = test[['resting_membrane_potential']].values
    y_test = test['group']
    model, best_params = train_best_model(X_train, y_train, n_splits=3)
    y_prob = model.predict_proba(X_test.reshape(-1, 1))
    auc = skmetrics.roc_auc_score(y_test, y_prob[:,-1])
    # plt.title(f'AUC: {auc:.2f}')

    # Return the calculated AUC
    return auc
