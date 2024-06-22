# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
##### ------------------------------------------------------------------- #####

def get_file_data(main_path:str, sep='_'):
    """
    Get file data in dataframe.

    Parameters
    ----------
    folder_path : str
    sep: str, string for splitting child dir names to categories

    Returns
    -------
    file_data : pd.DataFrame

    """
    
    # make lower string and path type
    main_path = os.path.normpath(main_path.lower())
    file_data = pd.DataFrame()
    
    # get only directories
    dirs = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
    
    # walk through all folders
    for folder in dirs:
        
        # get labchart file list
        full_path = os.path.join(main_path, folder)
        filelist = list(filter(lambda k: '.adicht' in k, os.listdir(full_path)))
            
        # get categories from folder
        temp_file_data = pd.DataFrame([folder.split(sep)]*len(filelist))
        
        # add folder and file
        temp_file_data.insert(0, 'file', filelist)
        temp_file_data.insert(0, 'folder_path', os.path.normcase(folder))

        file_data = file_data.append(temp_file_data, ignore_index=True)
                
    # convert data frame to lower case
    file_data = file_data.apply(lambda x: x.astype(str).str.lower())

    return file_data

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


