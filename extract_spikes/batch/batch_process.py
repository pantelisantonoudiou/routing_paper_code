# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import joblib
from joblib import Parallel, delayed
import contextlib
from batch.get_data import AdiGet
from  batch.current_clamp import Iclamp
##### ------------------------------------------------------------------- #####

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

def analyze_ic(raw_data, stim, fs, stim_type, wave, prominence, interpolation_factor=1,
               post_rheo_steps=-1, max_spikes_per_step=-1):
    """
    Extract outputs from different IC protocols per cell.

    Parameters
    ----------
    raw_data : array, voltage signal
    stim : array, input current
    fs : int, samping frequency (seconds)
    stim_type : str, name of comments in labchart
    wave : bool, True to get waveform for io stims
    prominence: float, spike prominence for detection
    interpolation_factor: int, interpolation factor, default = 1.
    post_rheo_steps: int, max number of IO steps to collect spikes post rheobase
    Default is -1 = collect from all steps
    max_spikes_per_step : int, number of max spikes to collect per step. 
    Default is -1 = collect all spikes.

    Returns
    -------
    output : dict, extracted data

    """
    
    # init current clamp object
    ic = Iclamp(fs, dist=1, prominence=prominence)
    
    if stim_type =='io':
        signal, i_amp, dur = ic.parse_io(raw_data, stim)
        locs = ic.spike_locs(signal)
    
    if (stim_type =='io') & (wave==False):
        spike_freq = np.array(ic.count_spikes(locs))/ (dur/fs)
        input_res = ic.get_input_resistance(signal, i_amp, ic.count_spikes(locs))
        rmp = ic.get_rmp(raw_data)
        output = {'spike_frequency':spike_freq, 'input_resistance': input_res,
                  'amp':i_amp, 'rmp':[rmp]*len(spike_freq)}
        
    if (stim_type == 'io') & (wave==True):
        waveform, amp, times = ic.select_waveforms(signal, i_amp, locs,
                                                   interpolation_factor=interpolation_factor,
                                                   post_rheo_steps=post_rheo_steps, 
                                                   max_spikes_per_step=max_spikes_per_step)

        if waveform:
            output = {'mV':waveform, 'amp':amp, 'time': times}
        else:
            output ={}
 
    if stim_type == 'rh':
        irheo = ic.get_rheobase(raw_data, stim)
        output = {'rheobase':[irheo]}
        
    if stim_type == 'sch':
        spike_count, freqs = ic.get_short_chirp(raw_data, stim, window=0.25)
        output = {'spike_count':spike_count, 'freq':freqs}

    return output


class BatchProcess():
    """
    Class for batch processing of single cell recordings
    """
    
    def __init__(self, main_path, index, data_ch, stim_ch, stim_corection, njobs=1):
        """

        Parameters
        ----------
        main_path : str, parent path
        index : TYPE
        data_ch : int, labchart channel corresponding to voltage data
        stim_ch : int, labchart channel corresponding to current stim
        stim_corection : int, to scale labchart stim to nA
        njobs : int, number of processes. The default is 1.

        Returns
        -------
        None.

        """
        self.main_path = main_path
        self.index = index[index.accepted == 1] # get only cells accepted from user verification
        self.data_ch = data_ch
        self.stim_ch = stim_ch
        self.stim_corection = stim_corection
        self.max_jobs = int(multiprocessing.cpu_count())-2
        
        # limit njobs within functional range
        self.njobs = njobs
        if self.njobs < 0:
            self.njobs = 1
        if self.njobs > self.max_jobs:
            self.njobs = self.max_jobs
            
    def extract_data(self, idx, row):
        """
        Load data and extract parameters from one cell.

        Parameters
        ----------
        idx : int, index from self.index dataframe.
        row : Pandas Series, row from self.index dataframe.

        Returns
        -------
        output : Pandas Df, with extracted parameters for one cell

        """
        # get data (convert stim to pA, signal already in mV)
        file_path = os.path.join(self.main_path, row.folder_path, row.file)
        raw_data, stim, fs = AdiGet(file_path, self.stim_type).get_data(data_ch=self.data_ch, stim_ch=self.stim_ch)
        if not np.any(raw_data):
            raise Exception('--> Could not read data from:' + file_path)
        stim *= self.stim_corection
        
        # extract data
        output = analyze_ic(raw_data, stim, fs, stim_type=self.stim_type,
                            wave=self.wave, prominence=row.threshold,
                            interpolation_factor = self.interpolation_factor,
                            post_rheo_steps=self.post_rheo_steps,
                            max_spikes_per_step=self.max_spikes_per_step)
        
        if output:
            output.update({'id':np.repeat(idx, len(output[list(output.keys())[0]]))})
        else:
            print('--> Could not extract parameters from: '+ file_path)
        return pd.DataFrame(data=output)
            
    
    def all_cells(self, stim_type, wave=False, interpolation_factor=1,
                  post_rheo_steps=-1, max_spikes_per_step=-1):
        """
        Run analysis for all cells in self.index file.

        Parameters
        ----------
        stim_type: str, name of stimulus type from [io, rh, sch].
        wave : bool, True to get waveform for io stims.
        interpolation_factor: int, interpolation factor, default = 1.
        post_rheo_steps: int, max number of IO steps to collect spikes post rheobase
        Default is -1 = collect from all steps
        max_spikes_per_step : int, number of max spikes to collect per step. 
            Default is -1 = collect all spikes.

        Returns
        -------
        data : Pandas Df, with extracted parameters
        """
        
        # pass parameters to object
        self.stim_type = stim_type
        self.wave = wave
        self.interpolation_factor = interpolation_factor
        self.post_rheo_steps = post_rheo_steps
        self.max_spikes_per_step = max_spikes_per_step
        
        # single processing
        if self.njobs==1:
            df_list = []
            for idx, row in tqdm(self.index.iterrows(), desc='Progress:', total=len(self.index)):
                output = self.extract_data(idx, row)
                df_list.append(output)
                
        # multi processing
        else:
            with tqdm_joblib(tqdm(desc='Progress:', total=len(self.index))) as progress_bar:  # NOQA
                df_list = Parallel(n_jobs=self.njobs, backend='loky')(delayed(self.extract_data)(idx, row) for idx, row in self.index.iterrows())
       
        # concatenate list and join with index
        df = pd.concat(df_list, axis=0)
        data = self.index.join(df.set_index('id')).reset_index()
        
        return data








    
