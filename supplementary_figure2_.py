# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extract_spikes.batch.batch_process import BatchProcess
from stats.outlier_removal import tukeys_fences
from plots.radar_plot import create_radar_plot
##### ------------------------------------------------------------------- #####


### =========================== Project settings ================================
main_path = '' # path to folder
stim_corection_factor = 1000
data_ch = 0
stim_ch = 1
stim_type = 'io'
njobs = 19
### =============================================================================

# investigate effect of increasing inputs to spike waveform
index = pd.read_csv(os.path.join(main_path, 'index_io.csv'))
batch = BatchProcess(main_path, index, data_ch, stim_ch, stim_corection_factor, njobs,)
data = batch.all_cells(stim_type, wave=False)
file_data = pd.DataFrame(data.groupby(by=['file'])[['spike_frequency']].max()).reset_index()
_, outliers = tukeys_fences(file_data['spike_frequency'],k=3)
    
post_rheobase_step = [3, 5, 7, -1]
max_spikes_per_step = [2, 3, 5, 7, -1]
comb = list(itertools.product(post_rheobase_step, max_spikes_per_step))
df_list = []
for rheo_step, max_step in comb:
    wave_data = batch.all_cells(stim_type, wave=True, post_rheo_steps=rheo_step, max_spikes_per_step=max_step, interpolation_factor=10)
    wave_data = wave_data.rename(columns={'0':'treatment', '1':'projection'})
    wave_data = pd.DataFrame(wave_data.groupby(by=['treatment', 'projection', 'file', 'time'])[['mV']].mean()).reset_index()
    excluded_cells = file_data['file'][file_data['spike_frequency']>=outliers.min()]
    wave_data = wave_data[~wave_data['file'].isin(excluded_cells)]
    wave_data = wave_data[wave_data['projection'].isin(['nonspecific'])]
    wave_data['steps_post_rheo'] = rheo_step
    wave_data['max_spikes_per_step'] = max_step
    df_list.append(wave_data)
all_wave_data = pd.concat(df_list).reset_index()
sns.relplot(data=all_wave_data, x='time', y='mV', hue='steps_post_rheo', kind='line',
            estimator=np.mean, errorbar='se', legend= 'full', col='max_spikes_per_step')

# load data and select conditions
properties = pd.read_csv(os.path.join('clean_data', 'all_properties.csv'))
properties = properties[ (properties['projection'].isin(['nonspecific']))]

# Convert to zscores
df = properties.copy()
columns = df.select_dtypes(include=['int', 'float']).columns
df[columns] = (df[columns] - df[columns].mean())/df[columns].std(ddof=0)

# plot the relative sdev of each variable when compared to controls
df_std = df.groupby(by=['treatment', 'projection']).std(numeric_only=True).reset_index()
df_std[columns] = df_std[columns].values/ df_std[columns][df_std['treatment'] == 'control'].values
df_std = df_std[df_std['treatment'] != 'control']
df_melt = df_std.melt(id_vars=['treatment','projection'])
df_melt = df_melt.rename(columns={'value': 'standard_deviation'})
num_cols = df_std.select_dtypes(include=['int', 'float']).columns.tolist()
df_std = df_std[df_std['treatment'] != 'control']
data_tuples = [(row['treatment'], [row[col] for col in num_cols]) for idx, row in df_std.iterrows()]
create_radar_plot(data_tuples, num_cols, colors=['#D6AFAE', '#83C18F'])

# plot the distribution of all variables
melted_df = pd.melt(df, id_vars=['treatment', 'projection', 'cell_id'])
hue_palette = dict(zip(melted_df['treatment'].unique(), ['#8EC0CC', '#D6AFAE', '#83C18F']))
g = sns.FacetGrid(melted_df, col='variable', col_wrap=4, hue='treatment',sharey=False,sharex=False, palette=hue_palette)
g.map(sns.kdeplot, 'value', common_norm=True, fill=True, bw_adjust=.8)
g.set_axis_labels('', '')
g.fig.tight_layout()
g.add_legend()









