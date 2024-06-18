# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plots.percent_spike import get_percent_spike_io
from plots.plots_with_stats import multi_group_comparison_plot
from sklearn.impute import KNNImputer
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
##### ------------------------------------------------------------------- #####

# settings
categories = ['cell_id', 'treatment']
treatment = ['control', 'cus', 'enriched-environment']
projections = ['nonspecific']
palette = ['#8EC0CC', '#D6AFAE', '#83C18F',]
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

# 1) plot io
sns.relplot(data=io_data, x='amp', y='spike_frequency',  marker='o', hue='treatment', kind='line', errorbar='se',palette=palette)

# 2) plot percent spike
df_percent_spike = get_percent_spike_io(io_data, ['treatment', 'amp'])
sns.relplot(data=df_percent_spike, x='amp', y='percent_spike', hue='treatment', marker='o', kind='line',palette=palette)
percent_spike_at_20 = df_percent_spike[df_percent_spike['amp'] == 30]

# 3) plot basic io properties and save stats
io_results = multi_group_comparison_plot(properties[categories+io_columns], 'treatment', io_columns, n_cols=4, palette=palette)
io_results.to_csv(os.path.join('stats', 'nonspecific_io_results.csv'), index=False, encoding='utf-8-sig')

# 4) plot waveform
sns.relplot(data=wave_data, x='time', y='mV', hue='treatment',kind='line', estimator=np.mean, errorbar='se', palette=palette)

# 5) plot wave properties and save stats
wave_results = multi_group_comparison_plot(properties[categories+wave_columns], 'treatment', wave_columns, n_cols=3, palette=palette)
wave_results.to_csv(os.path.join('stats', 'nonspecific_wave_results.csv'), index=False, encoding='utf-8-sig')

# 6) plot resting membrane potential distribution
sns.displot(data=properties, x='resting_membrane_potential', hue='treatment', kind='kde',fill=True, bw_adjust=1, palette=palette)

# 7) plot cells clustered based on control mode
# separate cells based on resting membrane potential
split_threshold = properties['resting_membrane_potential'][properties['treatment'] == 'control'].mode()[0]
properties['cluster'] = ''
depolarized_cells = properties.loc[properties['resting_membrane_potential'] >= (split_threshold), 'cell_id']
hyperpolarized_cells = properties.loc[properties['resting_membrane_potential'] < (split_threshold), 'cell_id']
properties.loc[properties['cell_id'].isin(depolarized_cells), 'cluster'] = 'depolarized-'
properties.loc[properties['cell_id'].isin(hyperpolarized_cells), 'cluster'] = 'hyperpolarized-'
properties.loc[properties['treatment'].isin(['control']), 'cluster'] = ''
properties['cluster']  = properties['cluster'] + properties['treatment']
plt.figure()
g = sns.kdeplot(data=properties, x='resting_membrane_potential', hue='cluster', bw_adjust=1, fill=True,)
g.axes.axvline(split_threshold)

### 8) Scatter plot RMP vs Imputed Rheobase ###
# separate cells based on resting membrane potential
properties.loc[properties['cell_id'].isin(depolarized_cells), 'cluster'] = 'depolarized'
properties.loc[properties['cell_id'].isin(hyperpolarized_cells), 'cluster'] = 'hyperpolarized'
properties.loc[properties['treatment'].isin(['control']), 'cluster'] = 'control'

# impute rheobase based on RMP
imputer = KNNImputer(n_neighbors=2, weights="uniform")
properties[['resting_membrane_potential', 'rheobase']] = imputer.fit_transform(properties[['resting_membrane_potential', 'rheobase']])

# plot control cells
plt.figure()
control = properties[properties['treatment'].isin(['control'])]
plt.scatter(control['resting_membrane_potential'], control['rheobase'], linewidths=2, edgecolor='white', s=100, label='control')

# plot cus and ee, depolarized and hyperpolarized clusters
cus_ee = properties[properties['treatment'].isin(['control']) == False]
symbols = {'cus':'x', 'enriched-environment': 's'}
colors = {'hyperpolarized':'black', 'depolarized':'orange'}
for conds, df in cus_ee.groupby(['treatment', 'cluster']):
    treatment, cluster = conds
    plt.scatter(df['resting_membrane_potential'], df['rheobase'], linewidths=2, edgecolor='white', s=100, label=conds,
                marker=symbols[treatment], color=colors[cluster])
plt.legend()
plt.axvline(split_threshold)
plt.xlabel('RMP')
plt.ylabel('Rheobase')
