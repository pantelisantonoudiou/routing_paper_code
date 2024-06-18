# -*- coding: utf-8 -*-


##### ----------------------------- IMPORTS ----------------------------- #####
import numpy as np
import pandas as pd
##### ------------------------------------------------------------------- #####

def get_percent_spike_io(io_data, conditions):
    """
    Calculate the percentage of cells spiking per IO step for each condition.

    Args:
        io_data (pandas.DataFrame): Input data containing information about IO steps and spike frequency.
        conditions (list): List of condition columns used for grouping the data.

    Returns:
        pandas.DataFrame: DataFrame containing the percentage of cells spiking per IO step for each condition.
    """
    # count number of cells spiking per IO step for each condition
    spike_portion = []
    for conds, df in io_data.groupby(by=conditions):
        temp_df = pd.DataFrame(df[conditions].iloc[[0]])
        temp_df['percent_spike'] = 100 * np.sum(df['spike_frequency'] > 2) / len(df)
        spike_portion.append(temp_df)
    spike_portion = pd.concat(spike_portion)
    return spike_portion
