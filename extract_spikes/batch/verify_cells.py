### ---------------------- IMPORTS ---------------------- ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from batch.get_index import get_file_data
from scipy.signal import find_peaks
### ----------------------------------------------------- ###


class matplotGui:
    """
        Matplotlib GUI for user seizure verification.
    """
    
    ind = 0 # set internal counter
       
    def __init__(self, main_path, stim_type='io', data_ch=0, stim_ch=1,
                 prominence=30, one_block=False):
        
        # get settings
        self.main_path = main_path
        self.data_ch = data_ch
        self.stim_ch = stim_ch
        self.stim_type = stim_type
        
        if one_block:
            from batch.get_data_one_block_no_com import AdiGet
        else:
            from batch.get_data import AdiGet
        self.AdiGet = AdiGet
        
        # get index file
        self.index_path = os.path.join(main_path, 'index_'+ stim_type + '.csv')
        if os.path.exists(self.index_path):
            self.index_df = pd.read_csv(self.index_path)
        else:
            self.index_df = get_file_data(main_path)
        
        # if first time verifying initialize values    
        if 'accepted' not in self.index_df.columns:
            self.index_df.insert(0, 'accepted', -1)
            self.index_df.insert(0, 'threshold', prominence)
        else:
            self.index_df['threshold']= prominence
            
        # wait time after plot
        self.wait_time = 0.05 # in seconds
        self.bcg_color = {-1:'w', 0:'salmon', 1:'palegreen'} 
        
        # create figure and axis
        self.fig, self.axs = plt.subplots(1, 1, sharex=False, figsize=(25, 15))

        # remove top and right axis
        self.axs.spines["top"].set_visible(False)
        self.axs.spines["right"].set_visible(False)
               
        # create first plot
        self.plot_data()
        
        # connect callbacks and add key legend 
        plt.subplots_adjust(bottom=0.15)
        self.fig.text(0.5, 0.04, '** Accept/Reject = a/r,         Previous/Next = \u2190/\u2192,         \
Increase/Decrease threshold = \u2191/\u2193 **\
\n**     Enter = Save, Esc = Close(no Save)      **' ,
                      ha="center", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))
        self.fig.canvas.callbacks.connect('key_press_event', self.keypress)
        self.fig.canvas.callbacks.connect('close_event', self.close_event)
        
        # disable x button
        win = plt.gcf().canvas.manager.window
        win.setWindowFlags(win.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        win.setWindowFlags(win.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        plt.show()

        
    def get_index(self):
        """
        Get dataframe index, reset when limits are exceeded
        
        Returns
        -------
        None.
        """

        # reset counter when limits are exceeded
        if self.ind >= len(self.index_df):
            self.ind = 0 
            
        elif self.ind < 0:
            self.ind = len(self.index_df)-1
       
        # set counter to internal counter
        self.i = self.ind
        
    def set_background_color(self, axis=[0,1]):
        """
        Set background color.

        Returns
        -------
        None.

        """
        clr = self.bcg_color[self.index_df['accepted'][self.i]]
        self.axs.set_facecolor(clr)
        
    def load_data(self):
        """
        Load labchart data.
        
        Returns
        -------
        raw_data : array
        fs : int
        file_name : str

        """
        
        # get data  (signal already in mV)
        self.get_index()
        row = self.index_df.loc[self.ind]
        file_path = os.path.join(self.main_path, row.folder_path, row.file)
        raw_data, _, fs = self.AdiGet(file_path, self.stim_type).get_data(
            data_ch=self.data_ch, stim_ch=self.stim_ch)
        return raw_data, fs, row.file
        

    def plot_data(self, **kwargs):
        """
        Plot detected spikes
        """
        
        # get file  and load data
        raw_data, fs, file_name = self.load_data()
        self.axs.clear()
        
        # check if data was accepted
        if raw_data is None:
            self.index_df.at[self.ind, 'accepted'] = 0
        else:

            # detect spikes
            spike_amp = self.index_df['threshold'][self.ind]
            spike_locs, _ = find_peaks(raw_data, prominence=spike_amp, distance=fs*1e-3, wlen=int(fs/10))
    
            # plot  data
            t = np.arange(0, raw_data.shape[0], 1) / fs
            threshold = 'Threshold = ' + str(self.index_df['threshold'][self.ind])
            self.axs.plot(t, raw_data, color='black', label=threshold)
            self.axs.plot(t[spike_locs], raw_data[spike_locs], linestyle='None',
                          marker='o', color='darkmagenta')
            self.axs.legend(loc='upper right')
        
        # format graphs
        self.fig.suptitle(str(self.ind+1) + ' of ' +  str(len(self.index_df)+1) + ' cells | ID = ' + file_name, fontsize=22)
        self.set_background_color()
        self.axs.set_xlabel('Time (Seconds)')
        self.axs.set_ylabel('Amp. (mV)')
        self.fig.canvas.draw()

            
    def save_idx(self):
        """
        Saves accepted PSD index and mat files.

        Returns
        -------
        None.
        """

        # check if all cells were verified
        if np.any(self.index_df['accepted'] == -1):
            print('\n****** Some cells were not verified ******\n')
            
        # store index csv
        self.index_df.to_csv(self.index_path, index=False)
        print('Cells were verified and index was saved to:', self.index_path)  
    
    
    
    ## ------  Cross press ------ ## 
    def close_event(self, event):
        plt.close()

    ## ------  Keyboard press ------ ##     
    def keypress(self, event):
        # print(event.key)
        if event.key == 'right':
            self.ind += 1 # add one to class index
            self.plot_data() # plot
            
        if event.key == 'left':
            self.ind -= 1 # subtract one to class index
            self.plot_data() # plot
        
        if event.key == 'up':
            self.index_df.at[self.ind, 'threshold'] = self.index_df['threshold'][self.ind] + 5
            self.plot_data() # plot
            
        if event.key == 'down':
            self.index_df.at[self.ind, 'threshold'] = self.index_df['threshold'][self.ind] - 5
            self.plot_data() # plot

        if event.key == 'a':
           # set values to arrays
           self.index_df.at[self.i, 'accepted'] = 1
                    
           # draw and pause for user visualization
           self.set_background_color()
           plt.draw()
           plt.pause(self.wait_time)
           
           # plot next event
           self.ind += 1 # add one to class index
           self.plot_data() # plot
          
        if event.key == 'r':
            # set values to arrays
            self.index_df.at[self.i, 'accepted'] = 0
            
            # draw and pause for user visualization
            self.set_background_color()
            plt.draw()
            plt.pause(self.wait_time)
            
            # plot next event
            self.ind += 1 # add one to class index
            self.plot_data() # plot
            
        if event.key == 'ctrl+a':
           # set values to arrays
           self.index_df.at[:, 'accepted'] = 1
           
           # draw and pause for user visualization
           self.set_background_color()
           plt.draw()
           plt.pause(self.wait_time)

        if event.key == 'ctrl+r':
            # set values to arrays
            self.index_df.at[:, 'accepted'] = 0
            
            # draw and pause for user visualization
            self.set_background_color()
            plt.draw()
            plt.pause(self.wait_time)
            
        if event.key == 'escape':
              plt.close()
        
        if event.key == 'enter':
            self.save_idx()
            plt.close() # trigger close callback
























