# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
##### ------------------------------------------------------------------- #####

def nonspecific_to_projections(properties, condition):
    """
    Assigns clusters and groups to cells based on their resting membrane potential and projections.

    Parameters:
        properties (dict): A dictionary containing cell properties.

    Returns:
        dict: The updated properties dictionary with assigned clusters and groups.
    """
    
    # Create copy
    properties = properties.copy()
    
    # Find the mode of the resting membrane potential for cells with 'control' treatment
    split_threshold = properties['resting_membrane_potential'][properties['treatment'] == 'control'].mode()[0]
    properties = properties[properties['treatment'].isin([condition])]
    properties['cluster'] = ''
    depolarized_cells = properties.loc[properties['resting_membrane_potential'] >= (split_threshold), 'cell_id']
    hyperpolarized_cells = properties.loc[properties['resting_membrane_potential'] < (split_threshold), 'cell_id']
    properties.loc[properties['cell_id'].isin(depolarized_cells), 'cluster'] = 'nonspecific-depolarized'
    properties.loc[properties['cell_id'].isin(hyperpolarized_cells), 'cluster'] = 'nonspecific-hyperpolarized'
    properties.loc[properties['projection'].isin(['bnst']), 'cluster'] = 'bnst'
    properties.loc[properties['projection'].isin(['nacc']), 'cluster'] = 'nacc'
    properties['group'] = ''
    
    bnst_vm = properties['resting_membrane_potential'][properties['projection'] == 'bnst'].mean()
    nacc_vm = properties['resting_membrane_potential'][properties['projection'] == 'nacc'].mean()
    if bnst_vm > nacc_vm:
        properties.loc[properties['cluster'].isin(['bnst']), 'group'] = 'depolarized'
        properties.loc[properties['cluster'].isin(['nacc']), 'group'] = 'hyperpolarized'
    else:
        properties.loc[properties['cluster'].isin(['bnst']), 'group'] = 'hyperpolarized'
        properties.loc[properties['cluster'].isin(['nacc']), 'group'] = 'depolarized'
        
    properties.loc[properties['cluster'].isin(['nonspecific-depolarized']), 'group'] = 'depolarized'
    properties.loc[properties['cluster'].isin(['nonspecific-hyperpolarized']), 'group'] = 'hyperpolarized'

    # Create a categorical plot using seaborn
    sns.catplot(data=properties, x='group', y='resting_membrane_potential', kind='box', hue='projection')

    # Return the updated properties dictionary
    return properties
