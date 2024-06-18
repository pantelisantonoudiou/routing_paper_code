# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plots.percent_spike import get_percent_spike_io
from plots.plots_with_stats import group_comparison_plot
from plots.boxplot_split import nonspecific_to_projections
from plots.decision_boundary_scatter import scatter_contour
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
##### ------------------------------------------------------------------- #####

# settings
paradigm = 'enriched-environment'
categories = ['cell_id', 'projection']
treatment = [paradigm]
projections = ['bnst', 'nacc']
palette = [ '#D1A5B5','#A7A9AC', ]
io_columns = ['fr_at_20_percent_input', 'fr_at_40_percent_input', 'fr_at_60_percent_input', 'fr_at_max_input', 
              'i_amp_at_half_max_fr', 'input_resistance', 'io_slope', 'max_firing_rate', 'resting_membrane_potential', 'rheobase']
wave_columns = ['ap_peak', 'ahp', 'peak_to_trough', 'threshold', 'rise_time', 'half_width']

# load data and select conditions
io_data = pd.read_csv(os.path.join('clean_data', 'io_data.csv'))
wave_data = pd.read_csv(os.path.join('clean_data', 'wave_data.csv'))
properties = pd.read_csv(os.path.join('clean_data', 'all_properties.csv'))
io_data = io_data[ (io_data['treatment'].isin(treatment)) & (io_data['projection'].isin(projections))]
wave_data = wave_data[ (wave_data['treatment'].isin(treatment)) & (wave_data['projection'].isin(projections))]
properties = properties[ (properties['treatment'].isin(treatment)) & (properties['projection'].isin(projections))]

# plot summary io
sns.relplot(data=io_data, x='amp', y='spike_frequency',  marker='o', hue='projection', kind='line', errorbar='se',palette=palette)

# plot percent spike
df_percent_spike = get_percent_spike_io(io_data, ['projection', 'amp'])
sns.relplot(data=df_percent_spike, x='amp', y='percent_spike', hue='projection', marker='o', kind='line',palette=palette)

# plot basic io properties and save stats
io_results = group_comparison_plot(properties[categories+io_columns], 'projection', io_columns, n_cols=4, palette=palette)
io_results.to_csv(os.path.join('stats', paradigm + '_projections_io_results.csv'), index=False, encoding='utf-8-sig')

# plot waveform
sns.relplot(data=wave_data, x='time', y='mV', hue='projection',kind='line', estimator=np.mean, errorbar='se', palette=palette)

# plot wave properties and save stats
wave_results = group_comparison_plot(properties[categories+wave_columns], 'projection', wave_columns, n_cols=3, palette=palette)
wave_results.to_csv(os.path.join('stats', paradigm + '_projections_wave_results.csv'), index=False, encoding='utf-8-sig')

# get depolarized and hyperpolarized and compare to projectors
properties = pd.read_csv(os.path.join('clean_data', 'all_properties.csv'))
properties = properties[ (properties['treatment'].isin( [paradigm]+['control']))]
split_properties = nonspecific_to_projections(properties, paradigm)

# match group labels and plot model decision boundary
split_properties.loc[split_properties['cluster'].isin(['nonspecific-depolarized']), 'group'] = 1
split_properties.loc[split_properties['cluster'].isin(['nonspecific-hyperpolarized']), 'group'] = 0
split_properties.loc[split_properties['cluster'].isin(['nacc']), 'group'] = 1
split_properties.loc[split_properties['cluster'].isin(['bnst']), 'group'] = 0
split_properties['group'] = split_properties['group'].astype(int).copy()
scatter_contour(split_properties, mapping={0:'#D1A5B5', 1:'#A7A9AC'}, bounds=5)
