# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
from batch.batch_process import BatchProcess
from properties import extract
from stats.outlier_removal import tukeys_fences
##### ------------------------------------------------------------------- #####

### =========================== Project settings ================================
main_path = '' # path to data folder
stim_corection_factor = 1000
data_ch = 0
stim_ch = 1
stim_type = 'io'
njobs = 19
categories = {'file':'cell_id', 'treatment':'treatment', 'projection':'projection'}
### =============================================================================

# # ### 1) create index files and verify spike detection across cells
# # # # =============================================================================
from batch.verify_cells import matplotGui
obj = matplotGui(main_path, stim_type='io', prominence=50) 
# # # # =============================================================================

if __name__ == '__main__':
    # get IO data and rename category columns
    index = pd.read_csv(os.path.join(main_path, 'index_'+stim_type+'.csv'))
    batch = BatchProcess(main_path, index, data_ch, stim_ch, stim_corection_factor, njobs,)
    data = batch.all_cells(stim_type, wave=False)
    data = data.rename(columns={'0':'treatment', '1':'projection'})
    
    # quantify basic properties
    plot_data = pd.DataFrame(data.groupby(by=['treatment', 'projection','file'])[['input_resistance', 'rmp']].mean()).reset_index()
    plot_data.loc[plot_data['input_resistance'] < 0 , 'input_resistance'] = np.NaN
    basic_properties = extract.get_basic_properties(plot_data, categories)
    
    # exclude cells with less than 5 spikes or 10Hz (manual validation of IO plots) and outliers based on Tukey's outlier removal
    file_data = pd.DataFrame(data.groupby(by=['file'])[['spike_frequency']].max()).reset_index()
    _, outliers = tukeys_fences(file_data['spike_frequency'], k=3)
    if len(outliers)==0:
        outliers = np.array([file_data['spike_frequency'].max()*2])
    excluded_cells = file_data['file'][(file_data['spike_frequency']<=10) | (file_data['spike_frequency']>=outliers.min())]
    data = data[~data['file'].isin(excluded_cells)]
    
    # quantify IO properties
    plot_data = pd.DataFrame(data.groupby(by=['treatment', 'projection', 'file', 'amp'])[['spike_frequency']].mean()).reset_index()
    plot_data = plot_data[plot_data['amp'] > 0]
    io_properties = extract.get_io_properties(plot_data, categories, show_plot=True)
    io_cols = io_properties.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # save clean io data for plotting
    plot_data.to_csv(os.path.join('clean_data','io_data.csv'), index=False)
    
    # get waveform data
    wave_data = batch.all_cells(stim_type, wave=True, post_rheo_steps=3, max_spikes_per_step=3, interpolation_factor=10)
    wave_data = wave_data.rename(columns={'0':'treatment', '1':'projection'})
    excluded_cells = file_data['file'][file_data['spike_frequency']>=outliers.min()]
    wave_data = wave_data[~wave_data['file'].isin(excluded_cells)]
    plot_data = pd.DataFrame(wave_data.groupby(by=['treatment', 'projection', 'file', 'time'])['mV'].median()).reset_index()
    
    # save clean waveform data for plotting
    plot_data.to_csv(os.path.join('clean_data','wave_data.csv'), index=False)
    
    # quantify waveform properties
    wave_data = batch.all_cells(stim_type, wave=True, post_rheo_steps=3, max_spikes_per_step=3, interpolation_factor=1)
    wave_data = wave_data.rename(columns={'0':'treatment', '1':'projection'})
    excluded_cells = file_data['file'][file_data['spike_frequency']>=outliers.min()]
    wave_data = wave_data[~wave_data['file'].isin(excluded_cells)]
    plot_data = pd.DataFrame(wave_data.groupby(by=['treatment', 'projection', 'file', 'time'])['mV'].median()).reset_index()
    wave_properties = extract.get_waveform_properties(plot_data, categories, show_plot=True)
    wave_cols = wave_properties.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # save joined properties
    properties = basic_properties.set_index('cell_id').combine_first(io_properties.set_index('cell_id')).combine_first(wave_properties.set_index('cell_id')).reset_index()
    obj_cols = properties.select_dtypes(include=['object']).columns.tolist()
    num_cols = properties.select_dtypes(include=['int', 'float']).columns.tolist()
    properties = properties[obj_cols+num_cols]
    properties.to_csv(os.path.join('clean_data','all_properties.csv'), index=False)

