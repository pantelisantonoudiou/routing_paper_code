# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 09:03:03 2022

@author: bstone04
"""

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system
import shutil

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import pandas as pd
from scipy.signal import find_peaks, peak_prominences,peak_widths
from tqdm import trange
import itertools
import time
from datetime import date
from scipy import signal as sg
from pandas.api.types import is_numeric_dtype
from itertools import combinations
from itertools import permutations

#import package to parallelize processes
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

#Import simple statistical packages
import math
from scipy.linalg import eigh
from scipy.linalg import norm
import matplotlib as mpl
from scipy import stats
from scipy.stats import t
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statistics import median

#import packages for PCA
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.decomposition import PCA as skPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from statsmodels.multivariate.pca import PCA as smPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

#Import packages for GMM
from sklearn import mixture
from sklearn import metrics
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import davies_bouldin_score

#Import plotting tools
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
plt.rcParams["axes.grid"] = False
import seaborn as sns

#fun dataframe tool
from functools import reduce

#Windows sucks, so we have to hide GMM warning about memory leaks
import warnings
warnings.filterwarnings('ignore')
# =============================================================================
# =============================================================================
# #                         #DEFINE FUNCTIONS TO USE
# =============================================================================
# =============================================================================
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalizenan(data):
    norm = np.sqrt(np.nansum(np.square(data)))
    return data/norm

def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=int(0.8*(multiprocessing.cpu_count())))(delayed(func)(group)\
                            for name, group in  tqdm(dfGrouped,  position=0, leave=True, colour='cyan',\
                                desc = 'Computing instantaneous phase'))
    return pd.concat([x for x in retLst if len(x)>0])

def applyParallel_trace(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group)\
                            for name, group in  tqdm(dfGrouped,  position=0, leave=True, colour='cyan',\
                                desc = 'Extraction of valid traces'))
    return retLst


# create function that extracts the raw traces for cells of interest
def trace_clipping_v1(element,trace_df):
    
    if element['cond'].unique()[0]=='CUS':
        if element['restraint'].unique()[0] == 'pre':
            if element['week'].unique()[0] == 1:
                order = 1
            if element['week'].unique()[0] != 1:
                order = 3
        if element['restraint'].unique()[0] != 'pre':
            if element['week'].unique()[0] == 1:
                order = 2
            if element['week'].unique()[0] != 1:
                order = 4      
    if element['cond'].unique()[0]=='EE':
        if element['restraint'].unique()[0] == 'pre':
            if element['week'].unique()[0] == 1:
                order = 1
            if element['week'].unique()[0] != 1:
                order = 7
        if element['restraint'].unique()[0] != 'pre':
            if element['week'].unique()[0] == 1:
                order = 2
            if element['week'].unique()[0] != 1:
                order = 8       
    extracted = trace_df.loc[(trace_df['animal']==element['animal'].unique()[0])&\
        (trace_df['order']==order)&(trace_df['cell']==element['cell'].unique()[0])\
            &(trace_df['cond']==element['cond'].unique()[0])].reset_index(drop=True)
    extracted['applied_cluster'] = element['cluster_num'].unique()[0]
    #extracted['order_original'] = element['order_original'].unique()[0]
    return extracted

def trace_clipping(element, trace_df):
    cond = element['cond'].unique()[0];  restraint = element['restraint'].unique()[0]
    week = element['week'].unique()[0]

    orders = {('CUS', 'pre', 1): 1,('CUS', 'pre', 4): 7,('CUS', 'post', 1): 2,\
       ('CUS', 'post', 4): 8,('EE', 'pre', 1): 1,('EE', 'pre', 4): 7,
        ('EE', 'post', 1): 2,('EE', 'post', 4): 8}

    order = orders.get((cond, restraint, week))

    if order is None:
        # Handle the case when the combination is not found
        # Set a default value or raise an error, based on your requirement
        order = -1  # Default value
    
    trace_df['order'] = trace_df['order'].astype(int)
    mask = ((trace_df['animal'] == element['animal'].unique()[0])&(trace_df['order'] == order)\
            &(trace_df['cell'].isin(element['cell'].unique())) & (trace_df['cond'] == cond))

    extracted = trace_df.loc[mask].reset_index(drop=True)
    extracted['applied_cluster'] = ''
    for clust in element['cluster_num'].unique():
        extracted.loc[extracted['cell'].isin(element.loc[element['cluster_num']==clust]['cell']),\
                      'applied_cluster'] = clust
    return extracted

def clipping_par(element):
    out = trace_clipping(element, trace_df)
    return out

def applyParallel_synch(dfGrouped, func):
    retLst = Parallel(n_jobs=int(0.8*(multiprocessing.cpu_count())))(delayed(func)(group)\
                            for name, group in  tqdm(dfGrouped,  position=0, leave=True, colour='magenta',\
                                desc = 'Computing phase synchrony'))
    return pd.concat([x for x in retLst if len(x)>0])

def classifier_par(element):
    out = classify(element, valid_df)
    return out

# create function that processes the features of an event "spike"
# and calculates instantaneous phase
def classify(element,trace_df):
    
    extracted = trace_df.loc[(trace_df['animal']==element['animal'].unique()[0])&\
        (trace_df['fname']==element['fname'].unique()[0])\
            &(trace_df['cell']==element['cell'].unique()[0])&\
                (trace_df['order']==element['order'].unique()[0])]
                
    # add the cluster number
    extracted.insert(6,'cluster_num',element['applied_cluster'].unique()[0])
    
    # now the element becomes the trace file
    element = extracted
    
    if len(element.loc[element['peaks']==True]['time_sec'])>1:
        # calculate instantaneous phase
        times = element['time_sec'].unique()
        spikes = np.asarray(element.loc[element['peaks']==True]['time_sec'])
        spike_inds = np.concatenate([[0], [np.where(times==x)[0][0] for x in spikes],[len(times)]])
        k = []; k_1 = []; phase = []
        for i in range(1,len(times)):
            this_k = spike_inds[spike_inds < i][-1]
            this_k_1 = spike_inds[spike_inds > i][0]
            this_phase = (2*np.pi*(i - this_k)/(this_k_1 - this_k))+2*np.pi*this_k
            k.append(this_k)
            k_1.append(this_k_1)
            phase.append(this_phase)
    else:
        phase = []
        
    # create dataframe that has 
    full_df = []
    for i in range(len(element.loc[element['peaks']==True])-1):
        
        p_time = element.loc[element['peaks']==True].iloc[i]['min_tl']
        iei = element.loc[element['peaks']==True].iloc[i+1]['min_tl'] - p_time
        
        if i ==0:
            previous_end = 0
        else:
            previous_end = element.loc[element['peaks']==True].iloc[i-1]['min_tr']
            
        start = element.loc[element['peaks']==True].iloc[i]['min_tl']
        rf = element.loc[element['time_sec'].between(previous_end,start,inclusive='both')]

        if len(rf)>0:
            # create dataframe for these features
            dset_df = pd.DataFrame([iei],columns=['iei'])
            dset_df.insert(0,'animal',element['animal'].unique()[0])
            dset_df.insert(1,'cell',element['cell'].unique()[0])
            dset_df.insert(2,'fname',element['fname'].unique()[0])
            dset_df.insert(3,'order',element['order'].unique()[0])
            dset_df.insert(4,'cluster_num',element['applied_cluster'].unique()[0])
            dset_df.insert(5,'event',i+1)
            dset_df.insert(6,'amp',element.loc[element['peaks']==True].iloc[i]['filtsig'])
            dset_df.insert(8,'mean_rf',np.mean(rf['filtsig']))
            dset_df.insert(9,'rf',[np.asarray(rf['filtsig'])])
            dset_df.insert(10,'norm_intensity',abs((dset_df['amp']-np.mean(rf['filtsig']))/np.mean(rf['filtsig'])))
            dset_df.insert(11,'rise_t',element.loc[element['peaks']==True].iloc[i]['rise_t'])
            dset_df.insert(12,'decay_t',element.loc[element['peaks']==True].iloc[i]['decay_t'])
            dset_df['spike_time'] = element.loc[element['peaks']==True].iloc[i]['time']
            dset_df['spike_time_sec'] = element.loc[element['peaks']==True].iloc[i]['time_sec']
            dset_df.insert(15,'instant_phase',[np.asarray(phase)])
            dset_df.insert(0,'cond',element['cond'].unique()[0])
            full_df.append(dset_df)
            
    if len(full_df)>0:
        full_df = pd.concat(full_df)
        
    return full_df

def pairwise_phase_synchrony(element):
    
    # generate empty list for dataframe compiling
    phase_sync_out = []
    
    # iterate over rows and perform calculation by time
    for index, row in element.iterrows():
        calc = np.sqrt((np.mean(np.cos(row['phase_diff'][:]))**2+\
                       np.mean(np.sin(row['phase_diff'][:]))**2))

        ps_df = pd.DataFrame([calc],columns=['phase_synch'])
        for i,var,val in zip(range(0,8),['cond','animal', 'order', 'applied_cluster',\
                        'compared_cluster', 'type','cell1', 'cell2'],\
                [row['cond'],row['animal'],row['order'],row['applied_cluster'],\
                 row['compared_cluster'],row['type'],row['cell1'],row['cell2']]):
            ps_df.insert(i,var,val)
        phase_sync_out.append(ps_df)
    
    return pd.concat(phase_sync_out)


def phase_synch_par(element):
    out = pairwise_phase_synchrony(element)
    return out

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)

def ellipseplot(gmm,ax,means,covs,colors):
    for i, (c,cov,color) in enumerate(zip(means,covs,colors)):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[i][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[i]
        
        #v, w = eigh(gmm.covariances_)
        v, w = eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        ax.scatter(c[0], c[1], marker='$%d$' %(i+1), alpha=1,
                    s=100, edgecolor='k')
        
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(c, v[0]/2, v[1], 180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.2)
        ax.add_artist(ell)

# create slope function for BIC Gradient calcs
def slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)


def scheirer_ray_hare_test(df, group_var, dep_var, indep_vars, reg_folder, gmm_folder):
    os.chdir(reg_folder)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    # Drop nan rows
    df.dropna(inplace=True)
    
    # Create design matrix
    X = df[indep_vars]
    X = sm.add_constant(X)
    y = df[dep_var]
   
    # Add interaction terms to the design matrix
    interaction = X[indep_vars[0]]
    interaction.rename('*'.join(indep_vars),inplace=True)
    for i in np.arange(1,len(indep_vars)):
        interaction *= X[indep_vars[i]]

    X = pd.concat([X, interaction], axis=1)
    
    # Fit the multiple regression model
    model  = sm.OLS(y, X).fit()
    
    # Create residual plots and save
    fig = plt.figure(figsize=(8, 8))
    sm.graphics.plot_regress_exog(model, X.columns[-1], fig=fig)
    plt.suptitle('Regression plots for %s' %(dep_var),fontsize=16)
    
    if group_var == None:
        savename = 'regplot_%sway_%s.pdf' %(len(indep_vars),dep_var)
    else:
        savename = 'regplot_%sway_%s_%s_%s.pdf' %(len(indep_vars),\
                                group_var,df[group_var].unique()[0],dep_var)
        
    fig.savefig(savename, bbox_inches="tight")
    plt.close(fig)
    
    # Get the results of the regression
    results = model.summary2().tables[1]
   
    # Calculate the predicted values for the dependent variable
    y_pred = model .predict(X)
   
    # Calculate R-squared, MSE, RMSE, and MAE
    r2 = model.rsquared; mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse);  mae = mean_absolute_error(y, y_pred)
   
    # Return results in a dataframe
    results = pd.concat([results, pd.DataFrame({"R2": [r2], "MSE": [mse],
                    "RMSE": [rmse],"MAE": [mae], "df_model": model.df_model,"df_resid": model.df_resid})], axis=1)
    results.rename(columns={results.columns[3]:'pval'},inplace=True)
    results = results.reset_index()
    results.rename(columns={results.columns[0]:'comp'},inplace=True)
    results['comp']=results['comp'].astype(str)
    os.chdir(gmm_folder)
    return results

def create_radar_plot(data, variables,ax,ylim=(0,1),ypad=10,title=None, colors=['red','navajowhite']):
        """
        Creates a radar plot with the given data and variables.

        Args:
        - data: a list of tuples, where each tuple contains a label and a list of values
        - variables: a list of strings representing the variables to plot
        - title: a string representing the title of the plot (default: None)
        - colormap: a string representing the name of the matplotlib colormap to use (default: 'viridis')

        Returns:
        - None
        """

        # Calculate angles for each variable
        num_vars = len(variables)
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))

        # Plot data on the radar plot
        for d, color in zip(data, colors):
            label, values = d
            values = np.array(values)
            values = np.concatenate((values, [values[0]]))
            ax.plot(theta, values.T, color=color, label=label)
            ax.fill(theta, values.T, facecolor=color, alpha=0.25)

        # Plot the reference dotted line
        #ax.plot(theta, np.ones_like(theta), linestyle='--', linewidth=2, color='k', alpha=0.5)
        ax.plot(theta, np.zeros_like(theta), linestyle='--', linewidth=2, color='k', alpha=0.5)

        # Customize the plot
        ax.set_thetagrids(np.degrees(theta[:-1]), variables)
        ax.tick_params(pad=ypad)
        #ax.set_rgrids([1], angle=180, fontsize=10, color='k', alpha=0.5)
        #ax.set_yticklabels([])

        # Add a legend and a title
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fontsize=14, frameon=False)

        if title is not None:
            ax.set_title(title, size=20, color='k', y=1.05)
        
        # Set the scale
        ax.set_ylim(ylim[0],ylim[1])
        
        # Show the plot
        plt.show()

def diff_calc(arr1,arr2,iters,percent):
    
    # Determine common length of vectors
    sample_len = min(len(arr1),len(arr2))
        
    # Create an empty array to store the bootstrapped differences
    diffs = np.empty(iters)
    
    # Perform bootstrapping (randoming choosing 80% of data)
    for i in range(iters):
        bootstrapped_group1 = np.random.choice(arr1, size=int(sample_len*(percent/100)), replace = True)
        bootstrapped_group2 = np.random.choice(arr2, size=int(sample_len*(percent/100)), replace = True)
        diffs[i] = np.mean(bootstrapped_group2 -bootstrapped_group1)
    
    # Calculate the standard error
    error_diff = np.std(diffs)
        
    return np.mean(diffs),error_diff,len(arr1),len(arr2),diffs

def func1(row):
    if len(row)>1:      #ensures that we have a pre and post condition
        stable = row['clusters_%s' %(best)].nunique() ==1
        row['stable'] = stable

        # sort dataframe to appropriately grab Pre condition
        row.sort_values(by='order', ascending=False) 
        
        # indicate cluster pair, but put nan for post (for proper post-hoc counts)
        row['cluster_pair'] = ['_'.join([str(x+1) for x in np.asarray(row['cluster_num'])]),np.nan]
        
        # create copy of frame
        copied = row.copy()
        
        # for the stable cells (those that had same cluster pre/post)
        # normalize their features by dividing post by pre
        if stable:
            copied = row.iloc[0].copy()     #copy the Pre data
            copied[analysis_vars]= row.iloc[1].copy()[analysis_vars]/row.iloc[0].copy()[analysis_vars]
            copied = pd.DataFrame(copied).transpose()
        return copied
    
def func2(row):
    if len(row)>1:      #ensures that we have a pre and post condition
        stable = row['clusters_%s' %(best)].nunique() ==1
        row['stable'] = stable

        # sort dataframe to appropriately grab Pre condition
        row.sort_values(by='order', ascending=False) 
        
        # indicate cluster pair, but put nan for post (for proper post-hoc counts)
        row['cluster_pair'] = ['_'.join([str(x+1) for x in np.asarray(row['cluster_num'])]),np.nan]
        
        # create copy of frame
        copied = row.copy()
        
        # for the stable cells (those that had same cluster pre/post)
        # normalize their features by dividing post by pre
        if stable:
            copied = row.iloc[0].copy()     #copy the Pre data
            copied[analysis_vars]= row.iloc[1].copy()[analysis_vars]/row.iloc[0].copy()[analysis_vars]
            copied = pd.DataFrame(copied).transpose()
        else:
            # this will allow us to see how 'cluster-hopping' impacts variables
            copied = row.iloc[0].copy()     #copy the Pre data
            copied[analysis_vars]= row.iloc[1].copy()[analysis_vars]/row.iloc[0].copy()[analysis_vars]
            copied = pd.DataFrame(copied).transpose()
        return copied


def synch_norm(row):
   
    # create copy of frame
    copied = row.copy()
    
    # ensures that we normalize using dataframes that have only recording 1
    if '1' in copied.order.unique():      

        # normalize their features by dividing by week 1 (pre)
        copied['norm_phase_synch'] = copied['phase_synch']/copied.loc[copied['order']=='1']['phase_synch'].iloc[0]
        
    # normalize across weeks (shows how acute stress changes synchrony)
    copied['norm_week'] = ''
    for i in range(1,8,2):
        try:
            copied.loc[copied['order']==str(int(i)+1),'norm_week'] = \
                copied.loc[copied['order']==str(int(i)+1)]['phase_synch'].iloc[0]/\
                copied.loc[copied['order']==str(i)]['phase_synch'].iloc[0]
        except:
            pass
        
    # normalize to week 1 pre (shows how chronic stress changes synchrony)
    copied['norm_pre'] = ''; copied['norm_post'] = ''
    for i in sorted(copied.order.unique()):
        if int(i)%2==1:  #PRE
            try:
                copied.loc[copied['order']==i,'norm_pre']  = \
                    copied.loc[copied['order']==i]['phase_synch'].iloc[0]/\
                    copied.loc[copied['order']=='1']['phase_synch'].iloc[0]
            except:
                pass
        else:       #POST
            try:
                copied.loc[copied['order']==i,'norm_post']  = \
                    copied.loc[copied['order']==i]['phase_synch'].iloc[0]/\
                    copied.loc[copied['order']=='2']['phase_synch'].iloc[0]
            except:
                pass
    return copied

# =============================================================================
# =============================================================================
# #           #PARAMETER SET UP AND FILE LOADING/MERGING
# =============================================================================
# =============================================================================
#Have user indicate what directory the .csv files reside in
dir_folder = easygui.diropenbox(msg = 'Choose where the peak dectection .csv files are located.')
os.chdir(dir_folder)

#Look for the csv files in the directory
file_list = os.listdir('./')
file_names = []
for files in file_list:
    if files[-3:] == 'csv':
        file_names.append(files)

#locates the property files (will use as identity flagging set)
csv_files = easygui.multchoicebox(
        msg = 'Select the files you want to use for this analysis...', 
        choices = (sorted([x if 'peaks' in x else 'No other options found'for x in file_names ],reverse=True))) 

#load files
peak_df = []
for df in csv_files:
    tmp = pd.read_csv(df,sep=",",engine='python')
    if 'CUS' in df:
        cond = 'CUS'

    else:
        cond = 'EE'
        
    #Update values 
    tmp.loc[(tmp['order'].isin([1,2])),'week'] = 1
    tmp.loc[~(tmp['order'].isin([1,2])),'week'] = 4
    tmp['restraint'] = np.where(tmp['order'] % 2 == 0,'post','pre')
    tmp.insert(0,'cond',cond)
    peak_df.append(tmp)
    
peak_df = pd.concat(peak_df)

# Ensure exact same recording length, for each animal*session
rec_t = 10      #in minutes
peak_df = peak_df.loc[peak_df['time_sec']<=60*rec_t]

# =============================================================================
# =============================================================================
#                   TAKE THE AVERAGE ACROSS THE DATAFRAME
# =============================================================================
# =============================================================================
#establish metrics to analyze
analysis_vars = ['filtsig','auc', 'width', 'prominence','rise_t', 'decay_t','peak_total']
labels = ['Peak Height','AUC','Width','Rise Time','Decay Time','Number of Events']

#create groupby to obtain files to process per analysis/comparison type
mean_df = peak_df.groupby(['cond','animal','week','restraint','cell'])[analysis_vars].mean().reset_index()
mean_df['cat'] = [1 if y['cond'] =='CUS' else 0 for x,y in mean_df.iterrows()]
mean_df['catrestraint'] = [1 if y['restraint'] =='pre' else 2 for x,y in mean_df.iterrows()]

#ensure each cell has pre/post by week
cnts_pp = mean_df.groupby(['cond','animal','week','cell'])['restraint'].count().reset_index(name='cnt')
cnts_cells = mean_df.groupby(['cond','animal','week','restraint'])['cell'].nunique()

# =============================================================================
# =============================================================================
# #                  #CREATE DIRECTORIES FOR GMM
# =============================================================================
# =============================================================================
##create directory for outputs
os.chdir(dir_folder) #defaults to where the csvs are

#create parent plot folder
if os.path.isdir(os.path.join(dir_folder,'analyses_%s' %(time.strftime("%Y%m%d")))):
    shutil.rmtree(os.path.join(dir_folder,'analyses_%s' %(time.strftime("%Y%m%d"))))

os.mkdir('./analyses_%s' %(time.strftime("%Y%m%d")))
parent_folder = dir_folder+'/analyses_%s' %(time.strftime("%Y%m%d"))
os.chdir(parent_folder)

#create parent plot folder
if os.path.isdir(os.path.join(parent_folder,'Pre_GMM')):
    shutil.rmtree(os.path.join(parent_folder,'Pre_GMM' ))

os.mkdir('./Pre_GMM')
pgmm_folder = parent_folder+'/Pre_GMM'

#create parent plot folder
if os.path.isdir(os.path.join(parent_folder,'Post_GMM' )):
    shutil.rmtree(os.path.join(parent_folder,'Post_GMM'))

os.mkdir('./Post_GMM')
gmm_folder = parent_folder+'/Post_GMM' 

#create parent plot folder
os.chdir(pgmm_folder)
if os.path.isdir(os.path.join(pgmm_folder,'regression_3way')):
    shutil.rmtree(os.path.join(pgmm_folder,'regression_3way'))

os.mkdir('./regression_3way')
preg3_folder = pgmm_folder+'/regression_3way'

#create parent plot folder
if os.path.isdir(os.path.join(pgmm_folder,'regression_2way')):
    shutil.rmtree(os.path.join(pgmm_folder,'regression_2way'))

os.mkdir('./regression_2way')
preg2_folder = pgmm_folder+'/regression_2way'

# =============================================================================
# =============================================================================
#                    USE COUNTS AND PERFORM CHI-SQUARE
# =============================================================================
# =============================================================================
#look at proportions of cells across weeks
wk_cnts = []; chistats = []
for week in sorted(mean_df.week.unique()):
    cnts = mean_df.loc[mean_df['week']==week].groupby(['cond',\
            'restraint'])['cell'].count().reset_index(name='cnt').sort_values(by=['cond','restraint'], ascending=False)

    # perform contigency test to test whether treatment is independent of
    # week number (condition) among BLA cells
    pivot = cnts.pivot(index="cond", columns=["restraint"],values="cnt")
    
    # Two-sample chi-square test
    cont_table = np.asarray(cnts.pivot(index="cond", \
                        columns=["restraint"],values="cnt"))

    result_chi = stats.chi2_contingency(cont_table)    
    
    vals = '\n$\it{X}^{2}$(%i) = %s, $\it{p}$ = %s' %(result_chi[2],\
           np.round(result_chi[0],2),np.round(result_chi[1],2))
    chistats.append(vals)
        
# Create key for all plot
cnts_all = mean_df.groupby(['cond','week','restraint'])['cell'].count().reset_index(name='cnt')
cnts_all['key'] = cnts_all['week'].astype(int).astype(str)+'_'+cnts_all['restraint']

# Plot it
fig,ax = plt.subplots(figsize=(10,8))
g = sns.barplot(data=cnts_all,  x="key", y='cnt',\
                hue="cond",order=['1_pre','1_post','4_pre','4_post'],\
            hue_order=['CUS','EE'], palette=['grey','mediumseagreen'],lw=3, \
              errcolor="black",edgecolor="black", ax=ax)
ax.set_xticklabels(['Pre','Post','Pre','Post'])
ax.set_xlabel(''); ax.set_ylabel('Number of Cells',fontsize=14)
ax.text(0.27, 0.05, 'Week 1', fontsize=14, transform=plt.gcf().transFigure)
ax.text(0.2, 0.01, '%s' %(chistats[0]), fontsize=14, transform=plt.gcf().transFigure)
ax.text(0.67, 0.05, 'Week 4', fontsize=14, transform=plt.gcf().transFigure)
ax.text(0.6, 0.01, '%s' %(chistats[1]), fontsize=14, transform=plt.gcf().transFigure)
ax.legend(title='Condition',bbox_to_anchor=(1.12, 0.75))
ax.set_title('Condition impact on cell count (pre/post-Acute Stress)\nChi-square contigency test',y=1.05,fontsize=18)
plt.savefig('cell_count.pdf', transparent=False, bbox_inches="tight")
plt.close()

# Run 3way multiple regression, but do not plot
stats_3WAY_preGMM = []
for var in analysis_vars:
    # run multiple linear regression function
    reg_out = scheirer_ray_hare_test(mean_df,None,var,['week','catrestraint','cat'], preg3_folder, pgmm_folder)
    
    reg_out['comp'] = reg_out['comp'].astype(str)
    reg_out.insert(0,'test','linearregression')
    reg_out.insert(2,'variable',var)
    reg_out['sig'] = np.where(reg_out['pval']<0.05,True,False)
    stats_3WAY_preGMM.append(reg_out)
stats_3WAY_preGMM = pd.concat(stats_3WAY_preGMM)
stats_3WAY_preGMM.to_csv('threeway_mixedlinearmodel_fixedeffects.csv')

# Plot responses (to look at how condition changes responses)
# And run the statistics
stats_2WAY_preGMM = []; stats_1WAY_preGMM = []
for rest in ['pre','post']:
    pset = mean_df.loc[mean_df['restraint']==rest]
    stats_2WAY_preGMM = []; stats_1WAY_preGMM = []
    for var in analysis_vars:
        
        # run multiple linear regression function
        reg_out = scheirer_ray_hare_test(pset,None,var,['week','cat'], preg2_folder, pgmm_folder)
        
        reg_out['comp'] = reg_out['comp'].astype(str)
        reg_out.insert(0,'test','linearregression')
        reg_out.insert(1,'restraint',rest)
        reg_out.insert(2,'variable',var)
        reg_out['sig'] = np.where(reg_out['pval']<0.05,True,False)
        stats_2WAY_preGMM.append(reg_out)
        
        # Run one-way non-parametric test within week
        wk_colors =[]
        for week in [1,4]:
            # Run one way
            grouped = pset.loc[pset['week']==week].groupby(['cond'])[var]
            stat, p = stats.kruskal(*[grouped.get_group((i)) for i in grouped.groups])
            dof = len(grouped) - 1
            test = pd.DataFrame(np.expand_dims(np.asarray([dof,stat,p]),1).T,\
                                columns=['dof','stat','pval'])
            test.insert(0,'test','kruskal')
            test.insert(1,'variable',var)
            test.insert(2,'comp','cond')
            test.insert(3,'week',week)
            test.insert(4,'restraint',rest)
            stats_1WAY_preGMM.append(test)
            if p < 0.05:
                color = 'red'
            else:
                color= 'black'
            wk_colors.append(color)
    
        fig,ax = plt.subplots(figsize=(8,8))
        g = sns.barplot(data=pset,  x="week", y=var,\
                        hue="cond",order=[1,4],hue_order=['CUS','EE'],
                     palette=['grey','mediumseagreen'],lw=3,ci=68, \
                      errcolor="black",edgecolor="black", ax=ax)
        ax.set_ylabel('Value',fontsize=16)
        ax.set_xlabel('Week #',fontsize=16)
        for i,color in zip(range(2),wk_colors):
            ax.get_xticklabels()[i].set_color(color)
        ax.set_xticklabels([1,4],fontsize=14)
        compset = reg_out.loc[reg_out['comp']=='0']
        stat_vals = reg_out.loc[reg_out['comp']=='week*cat']
        ax.set_title('Condition impact on %s-acute stress: %s of cells' %(rest.title(),var)+\
                     '\n$\it{t}$(%i,%i) = %s, $\it{p}$ = %s, \u03B2=%s' %(compset['df_model'].iloc[0],\
                     compset['df_resid'].iloc[0],np.round(stat_vals['t'].iloc[0],2),"{:.2e}".format(stat_vals['pval'].iloc[0]),\
                         "{:.2e}".format(stat_vals['Coef.'].iloc[0]/stat_vals['Std.Err.'].iloc[0])),fontsize=18)
        ax.legend(title='Condition',bbox_to_anchor=(1.3, 0.75))
        plt.savefig('grouped_%s_%s.pdf' %(rest,var), transparent=False, bbox_inches="tight")
        plt.close()

stats_1WAY_preGMM = pd.concat(stats_1WAY_preGMM)
stats_2WAY_preGMM = pd.concat(stats_2WAY_preGMM)
stats_1WAY_preGMM.to_csv('oneway_kruskal.csv')
stats_2WAY_preGMM.to_csv('twoway_mixedlinearmodel_fixedeffects.csv')
    
# Calculate difference scores (condition-based)
diff_preGMM_df = []
for var in analysis_vars:
    # negative value = Post-acute restraint mean was higher
    for idx,df in mean_df.groupby(['cond','week']):
        
        # create x, y variables
        preset = df.loc[df['restraint']=='pre']
        postset = df.loc[df['restraint']=='post']
        
        # get sample size
        pre_n = len(preset); post_n = len(postset)
        
        # calculte pooled std and sems
        mean_diff = np.mean(postset[var])-np.mean(preset[var])
        std1 = np.std(preset[var],ddof=1); std2 = np.std(postset[var],ddof=1)
        pooled_std = np.sqrt(((len(preset[var])-1)*std1**2+(len(postset[var])-1)*std2**2)/\
                             (len(preset[var])+len(postset[var])-2))
        pooled_sem = pooled_std*np.sqrt((1/len(preset[var]))+(1/len(postset[var])))
        
        # run statistics
        t,p = stats.ttest_ind_from_stats(np.mean(preset[var]), std1, pre_n, np.mean(postset[var]), std2, post_n)
        out = pd.DataFrame(np.expand_dims(np.asarray([idx[0],idx[1],var,pre_n,\
                                        post_n,mean_diff,std1,std2,pooled_std,pooled_sem,t,p]),1).T,\
            columns=['cond','week','variable','n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p'])
        out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']] = \
            out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']].astype(float) 
        diff_preGMM_df.append(out)
        
diff_preGMM_df = pd.concat(diff_preGMM_df)

# perform ttest on pooled means/stds (within condition)
diff_preGMM_df_week = diff_preGMM_df.copy()

diff_preGMM_df_week['tval']=''; diff_preGMM_df_week['pval']=''
for idx,df in diff_preGMM_df_week.groupby(['variable','cond']):
    df['week'] = df['week'].astype(float)
    
    pooled_1 = df.loc[df['week']==1]['n_pre']+df.loc[df['week']==1]['n_post']
    pooled_4 = df.loc[df['week']==4]['n_pre']+df.loc[df['week']==4]['n_post']
    t,p = stats.ttest_ind_from_stats(df.loc[df['week']==1]['mean'], df.loc[df['week']==1]['pooled_std'],\
                        pooled_1,df.loc[df['week']==4]['mean'], df.loc[df['week']==4]['pooled_std'], pooled_4)
    diff_preGMM_df_week.loc[(diff_preGMM_df_week['variable']==idx[0])&(diff_preGMM_df_week['cond']==idx[1]),'tval'] = t
    diff_preGMM_df_week.loc[(diff_preGMM_df_week['variable']==idx[0])&(diff_preGMM_df_week['cond']==idx[1]),'pval'] = p
diff_preGMM_df_week[['tval','pval']] = diff_preGMM_df_week[['tval','pval']].astype(float)
diff_preGMM_df_week.to_csv('week_differences_preGMM_stats.csv')

# perform ttest on pooled means/stds
diff_preGMM_df['tval']=''; diff_preGMM_df['pval']=''
for idx,df in diff_preGMM_df.groupby(['variable','week']):

    pooled_cus = df.loc[df['cond']=='CUS']['n_pre']+df.loc[df['cond']=='CUS']['n_post']
    pooled_ee = df.loc[df['cond']=='EE']['n_pre']+df.loc[df['cond']=='EE']['n_post']
    t,p = stats.ttest_ind_from_stats(df.loc[df['cond']=='CUS']['mean'], df.loc[df['cond']=='CUS']['pooled_std'],\
                        pooled_cus,df.loc[df['cond']=='EE']['mean'], df.loc[df['cond']=='EE']['pooled_std'], pooled_ee)
    diff_preGMM_df.loc[(diff_preGMM_df['variable']==idx[0])&(diff_preGMM_df['week']==idx[1]),'tval'] = t
    diff_preGMM_df.loc[(diff_preGMM_df['variable']==idx[0])&(diff_preGMM_df['week']==idx[1]),'pval'] = p
diff_preGMM_df[['tval','pval']] = diff_preGMM_df[['tval','pval']].astype(float)
diff_preGMM_df.to_csv('week_differences_cond_stats.csv')

# perform ttest on pooled means/stds (within condition)
diff_preGMM_df_week = diff_preGMM_df.copy()

# save output
diff_preGMM_df.to_csv('cond_differences_preGMM.csv')

# set stastical variables
alpha = 0.05

# create dataframe for condition-induced changes
diff_preGMM_df_cond = []

# initiate figure
fig,axes = plt.subplots(figsize=(14,14),nrows=3,ncols=3)
for var,label,ax,cnt in zip(analysis_vars,['Peak Height', 'AUC', 'Width',\
            'Prominence','Rise Time', 'Decay Time', 'Number of Events'],\
                    axes.flatten(),range(len(analysis_vars))):
    # plot it
    g = sns.barplot(data=diff_preGMM_df.loc[(diff_preGMM_df['variable']==var)],ci=None,\
                x="week", y='mean',hue="cond", hue_order=['CUS','EE'],
                 palette=['grey','mediumseagreen'],lw=3, \
                  errcolor="black",edgecolor="black", ax=ax)
    
    # Add error bars (from bootstrap)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    sems = list(diff_preGMM_df.loc[(diff_preGMM_df['variable']==var)]["pooled_sem"])
    if np.where(np.isnan(y_coords)==True)[0]:
        tmp = np.zeros_like(x_coords)
        for i in np.where(np.isnan(y_coords)==True):
            tmp[i] = np.nan
            sems.insert(i[0],np.nan)
    ax.errorbar(x=x_coords, y=y_coords, yerr=sems, fmt="none", c= "k",lw=2)
    ax.set_title(label,fontsize=14,fontweight='bold')
    
    # extract sig vals
    sigs = [float(x) for x in diff_preGMM_df.loc[(diff_preGMM_df['variable']==var)&\
                              (diff_preGMM_df['cond']=='CUS')]['pval']]
    sigcolors = ['red' if x<=alpha else 'black' for x in sigs]
    
    # update y-label and x-labels and ticks
    ax.set_ylabel(''); ax.set_xlabel('')
    ax.set_xticklabels(['Week %i' %(x) for x in [1,4]],fontsize=14)
    [plt.setp(ax.get_xticklabels()[x], color=y) for x,y in zip(range(2),sigcolors) ]
    
    if var !='width':
        ax.get_legend().set_visible(False)
    else:
        ax.legend(title='Condition',bbox_to_anchor=(1, 0.75))
    if cnt in [0,3,6]:
        ax.set_ylabel('Change Score (post-pre)')

    for idx,df in diff_preGMM_df.loc[(diff_preGMM_df['variable']==var)].groupby('week'):
        
        # create x, y variables
        preset = df.loc[df['cond']=='CUS']
        postset = df.loc[df['cond']=='EE']
        
        # calculte pooled std and sems
        mean_diff = postset['mean']-preset['mean']
        std1 = preset['pooled_std']; std2 = postset['pooled_std']
        
        # get sample sizes
        pooled_ee = postset['n_pre']+postset['n_post']
        pooled_cus = preset['n_pre']+preset['n_post']
        
        pooled_std = np.sqrt(((pooled_cus-1)*std1**2+(pooled_ee-1)*std2**2)/\
                             (pooled_cus+pooled_ee-2))
        pooled_sem = pooled_std*np.sqrt((1/pooled_cus)+(1/pooled_ee))
        
        # run statistics
        t,p = stats.ttest_ind_from_stats(preset['mean'], std1, pooled_cus, postset['mean'], std2, pooled_ee)
        
        # create dataframe
        out = pd.DataFrame(np.expand_dims(np.asarray([idx[0],var,pooled_cus,\
                                        pooled_ee,mean_diff,std1,std2,pooled_std,pooled_sem,t,p]),1).T,\
            columns=['week','variable','n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p'])
        out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']] = \
            out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']].astype(float) 
        diff_preGMM_df_cond.append(out)

# Save the figure
plt.suptitle('Condition impact on calcium properties, raw', fontsize=15)
plt.savefig('grouped_raw_data.pdf', transparent=False, bbox_inches="tight")
plt.close()

# Concatenate frames
diff_preGMM_df_cond = pd.concat(diff_preGMM_df_cond)

# perform ttest on pooled means/stds
diff_preGMM_df_cond['tval']=''; diff_preGMM_df_cond['pval']=''
for idx,df in diff_preGMM_df_cond.groupby(['variable']):

    pooled_1 = df.loc[df['week']==str(1)]['n_pre']+df.loc[df['week']==str(1)]['n_post']
    pooled_4 = df.loc[df['week']==str(4)]['n_pre']+df.loc[df['week']==str(4)]['n_post']
    t,p = stats.ttest_ind_from_stats(df.loc[df['week']==str(1)]['mean'], df.loc[df['week']==str(1)]['pooled_std'],\
                        pooled_1,df.loc[df['week']==str(4)]['mean'], df.loc[df['week']==str(4)]['pooled_std'], pooled_4)
    diff_preGMM_df_cond.loc[(diff_preGMM_df_cond['variable']==idx),'tval'] = t
    diff_preGMM_df_cond.loc[(diff_preGMM_df_cond['variable']==idx),'pval'] = p
diff_preGMM_df_cond[['tval','pval']] = diff_preGMM_df_cond[['tval','pval']].astype(float)

# create x-positions
diff_preGMM_df_cond['xpos'] = ''
for x,var in zip(range(len(analysis_vars)),analysis_vars):
    diff_preGMM_df_cond.loc[(diff_preGMM_df_cond['variable']==var)&\
                            (diff_preGMM_df_cond['week']==str(1)),'xpos'] = x-0.15
    diff_preGMM_df_cond.loc[(diff_preGMM_df_cond['variable']==var)&\
                            (diff_preGMM_df_cond['week']==str(4)),'xpos'] = x+0.15

cond_sigs = ['orange' if x<0.05 else 'black' for x in np.asarray(diff_preGMM_df_cond['p'])]
sigs = ['red' if x<0.05 else 'black' for x in np.asarray(diff_preGMM_df_cond.loc[diff_preGMM_df_cond['week']==str(1)]['pval'])]

# plot it (all data)
fig, ax = plt.subplots(figsize=(10, 10))
ax.errorbar(diff_preGMM_df_cond['xpos'],diff_preGMM_df_cond['mean'],lw=3,alpha=0.6,\
            yerr = diff_preGMM_df_cond['pooled_sem'], ls = "None", ecolor=cond_sigs)
ax.scatter(diff_preGMM_df_cond['xpos'],diff_preGMM_df_cond['mean'],\
           s = 100, marker = "h", color = cond_sigs)
ax.set_xticklabels([0,'Peak Height', 'AUC', 'Width','Prominence',\
                    'Rise Time', 'Decay Time', 'Number of Events'],fontsize=14)
[plt.setp(ax.get_xticklabels()[x], color=y) for x,y in zip(range(len(analysis_vars)),sigs)]
# Save the figure
plt.suptitle('Condition impact on calcium properties,\nchange score (negative: CUS stronger AR difference', fontsize=15)
plt.savefig('grouped_changescore_data_v1.pdf', transparent=False, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(figsize=(27, 7),ncols=7)
for var,ax,colors,label,scolor in zip(analysis_vars,axes.flatten(),\
    [[cond_sigs[x],cond_sigs[x+1]] for x in range(len(analysis_vars))],\
        ['Peak Height', 'AUC', 'Width', 'Prominence',\
         'Rise Time', 'Decay Time', 'Number of Events'],sigs):
    dset = diff_preGMM_df_cond.loc[diff_preGMM_df_cond['variable']==var]
    ax.errorbar(range(2),dset['mean'],lw=3,alpha=0.6,\
                yerr = dset['pooled_sem'], ls = "None", ecolor=colors)
    ax.scatter(range(2),dset['mean'],\
               s = 100, marker = "h", color = colors)
    ax.set_xlim(-1,2)
    ax.set_title(label,fontsize=14,fontweight='bold',color=scolor)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Week 1', 'Week 4'],fontsize=13)
    
# Save the figure
plt.suptitle('Condition impact on calcium properties,\nchange score (negative: CUS stronger AR difference',\
             fontsize=15,y=1)
plt.savefig('grouped_changescore_data_v2.pdf', transparent=False, bbox_inches="tight")
plt.close()

# Create radar plots
properties = ['filtsig', 'auc', 'width','prominence', 'rise_t', 'decay_t', 'peak_total']
mean_radar_df = mean_df.groupby(['cond','week','restraint'])[properties].mean().reset_index()

# Normalize the data
mean_radar_z_df = mean_radar_df.copy()
mean_radar_z_df[properties] = (mean_radar_z_df[properties] - mean_radar_z_df[properties].mean()) / mean_radar_z_df[properties].std()

# Calculate data for radar plots
radar_data = []; radar_vals = []
for cond in mean_radar_z_df.cond.unique():
    dataset =[]
    for week in sorted(mean_radar_z_df.week.unique()):
        dset = mean_radar_z_df.loc[(mean_radar_z_df['cond']==cond)&(mean_radar_z_df['week']==week)]
        change_df = dset.loc[dset['restraint']=='post'].reset_index(drop=True)
        
        # normalize the values before subtracting
        change_df[properties]=change_df[properties]-dset.loc[dset['restraint']=='pre'].reset_index(drop=True)[properties]
        dataset.append((week,list(change_df[properties].iloc[0]))); radar_vals.append(list(change_df[properties].iloc[0]))
    radar_data.append(dataset); 

# Get mins and maxs to scale the figures appropriately
radar_min = min(np.concatenate([x for x in radar_vals]))
radar_max = max(np.concatenate([x for x in radar_vals]))

# Plot the data
fig,axes = plt.subplots(figsize=(10,10),subplot_kw={'projection': 'polar'},\
                        ncols=len(mean_radar_z_df.cond.unique()))
axes_list = axes.flatten(); cnt=0
for cond in mean_radar_z_df.cond.unique():
    dataset =[]
    for week in sorted(mean_radar_z_df.week.unique()):
        dset = mean_radar_z_df.loc[(mean_radar_z_df['cond']==cond)&(mean_radar_z_df['week']==week)]
        change_df = dset.loc[dset['restraint']=='post'].reset_index(drop=True)
        
        # normalize the values before subtracting
        change_df[properties]=change_df[properties]-dset.loc[dset['restraint']=='pre'].reset_index(drop=True)[properties]
        dataset.append((week,list(change_df[properties].iloc[0])))
    
    # Plot it
    create_radar_plot(dataset, properties, axes_list[cnt],ylim=(np.round(radar_min*1.1,2),\
                       np.round(radar_max*1.1,2)),ypad=20,title='%s' %(cond),colors=['gold','slateblue'])
    axes_list[cnt].set_yticklabels([])
    cnt+=1
fig.set_size_inches(18.5,10.5)
plt.savefig('radar_plots.pdf', transparent=False, bbox_inches="tight")
plt.close()

# =============================================================================
#                    Create waveforms from data
# =============================================================================
# The formula for this comes from this paper (CITE)
waveform_df = mean_df.groupby(['week','restraint','cond'])[analysis_vars].mean().reset_index()
waveform_df['time'] = ''; waveform_df['wave'] = ''
for idx,df in waveform_df.iterrows():
    
    trace_length = 5000     #in milisceconds
    peak_height,event_width,rise_t,decay_time,area,event_count = \
        df[['filtsig','width','rise_t', 'decay_t','auc', 'peak_total']]
        
    t = np.linspace(0, 12, 1000)
    signal = peak_height * np.exp(-t / rise_t) * (1-np.exp(-t /decay_time))

    signal *= area / np.sum(signal)

    # Store it
    waveform_df.iloc[idx,-2] = np.expand_dims(t,1)
    waveform_df.iloc[idx,-1] = np.expand_dims(signal,1)

# Save waveform dataframe
waveform_df.to_csv('waveform_df.csv')

# Plot the signals based on condition
cond_colors = ['black','salmon']
fig,axes= plt.subplots(figsize=(8,5),ncols=2,sharey=True,sharex=True)
axes_list = axes.flatten()
for idx,df in waveform_df.groupby(['week']):

    if idx == 1:
        col = 0
    else:
        col = 1
    
    inset = axes[col].inset_axes([0.6,0.6,0.35,0.35])
    g = sns.barplot(data = df, x = 'restraint',y='peak_total',\
                    order=['pre','post'],hue='cond',\
                    hue_order=sorted(df.cond.unique()),
                    palette=cond_colors,ax=inset,edgecolor="black")
    inset.legend().set_visible(False)
    inset.set_xticklabels(['Pre','Post'],fontsize=8)
    inset.set_xlabel('Restraint',fontsize=9)
    inset.set_ylabel('Total events',fontsize=9)
    inset.set_ylim(0,120)
    inset.tick_params(axis='both', which='major', labelsize=8)
    
    # create dotted bars, in post, to match the waveforms
    for i, bar in enumerate(inset.patches):
        bar.set_edgecolor(np.repeat(cond_colors,2)[i])
        if i % 2 != 0:
            bar.set_hatch('..')
            bar.set_facecolor('white')
        
    for cond,color in zip(sorted(df.cond.unique()),cond_colors):
        for restraint, lstyle in zip(['pre','post'],['-',':']):
            dset = df.loc[(df['restraint']==restraint)&(df['cond']==cond)]
            axes[col].plot(np.asarray(dset['time'].iloc[0]).T,\
                                  np.asarray(dset['wave'].iloc[0]).T,\
                color=color,ls=lstyle)
    axes[col].set_title('Week: %i' %(idx))
    
    # Formatting
    axes[0].set_ylabel('Normalized\n'+r'Fluorescence ($\Delta$F/F)',fontsize=13)
    axes[col].set_xlabel('Time (sec)',fontsize=13)
    axes[col].set_xlabel('Time (sec)',fontsize=13)
    
# Save the figure
plt.suptitle('Condition impact on calcium waveforms', fontsize=15)
plt.savefig('waveform_dynamics.pdf', transparent=False, bbox_inches="tight")
plt.close()

# =============================================================================
# =============================================================================
# =============================================================================
# #                  TAKE USER INPUT FOR DYNAMIC PROCESSING
# =============================================================================
# =============================================================================
# =============================================================================
# Ask the user for the initialization paramters
init_params = easygui.multenterbox(title='Processing parameters',
                                  msg = 'Indicate processing parameters for GMM algorithm.',\
                                  fields = ['Threshold of variance explained by PCA (%)',\
                                            'GMM confidence (0-1, 0.75 is high)'],\
                                  values = [95,0.70])
init_params = [float(x) for x in init_params]

# =============================================================================
# =============================================================================
# =============================================================================
#                    PERFORM ON ALL DATA AT W/O PCA
# =============================================================================
# =============================================================================
# =============================================================================
os.chdir(parent_folder)

# define min max scaler
scaler = MinMaxScaler(feature_range=(-1,1))

# transform data
scaled = scaler.fit_transform(np.asarray(mean_df.iloc[:,5:12]))

# change directories
os.chdir(gmm_folder)

#Set-up GMM dataframe (using normalized data)
X = scaled

#establish appropriate component comparisons
if X.shape[1] >= 7:
    comp_length = 7
if X.shape[1] < 7:
    comp_length = X.shape[1]-1

# Detail number of components
n_components = range(1, comp_length)

# Compute the BIC score for different number of components
fig,ax = plt.subplots(figsize=(6,5))
gmm_set=[]
for ctype in ['spherical', 'tied', 'diag', 'full']:
    bic_scores = []
    for n in n_components:
        
        gmm = mixture.GaussianMixture(n_components=n,\
                    covariance_type=ctype, random_state=0).fit(X)
        bic_scores.append(gmm.bic(X))
    
    # Choose the optimal number of components based on the elbow in the BIC score
    components = range(1, len(bic_scores))
    first_derivatives = np.diff(bic_scores)
    second_derivatives = np.diff(first_derivatives)
    
    elbow_idx = np.argmin(second_derivatives) + 1
    optimal_n_components = components[elbow_idx]
    gmm_set.append((ctype,optimal_n_components))
    
    # Plot the BIC score
    ax.plot(n_components, bic_scores, label = ctype, lw=3,alpha=0.6)
    ax.scatter(optimal_n_components,bic_scores[optimal_n_components-1],\
               linewidth=3,facecolors='none',s=250,edgecolor='black')
    
ax.set_xlabel('Number of components')
ax.set_ylabel('BIC score')
ax.legend(title='Covariance Type',loc=1)
ax.set_title('Optimal cluster number determination\nElbow Method with 2nd order derivative'+\
             '\non raw CA properties',fontsize=14, fontweight='bold',y = 1.05)
plt.savefig('GMM_cluster_determine_raw.pdf', bbox_inches="tight")
plt.close(fig)

#create list to store outputs
gmms = [];gmm_outs = [];gmm_centers = [];percents = []
fig,axes = plt.subplots(figsize=(10,10),nrows=2,ncols=2,sharey=True,sharex=True)
for ax,gmm_vals in zip(axes.flatten(),gmm_set):
    
    best = gmm_vals[1]
    bestmodel = gmm_vals[0]
    
    #create copy of PCs
    pca_df = pd.DataFrame(X,columns=['P%i' %(x+1) for x in range(X.shape[1])])
    
    #Run the GMM based on user input
    gmm_colors=['lightseagreen','magenta','darkblue','orange','rebeccapurple','crimson']
    gmm = mixture.GaussianMixture(n_components=best,
                    covariance_type=bestmodel, random_state=0)
    gmm.fit(pca_df)
    gmms.append(gmm)
    
    #add identifiers to dataframe
    pca_df['clusters_%s' %(best)] = pd.to_numeric(gmm.predict(pca_df))
    
    #calculate probability estimates of cluster assignment
    weights = gmm.predict_proba(X)
    for var,i in zip(['probest_%i_%i'%(best,x) for x in range(best)],range(best)):
        pca_df[var] = weights[:,i]
        
    #create flag to denote cluster assignment that does not meet user's input
    pca_df['%s_crit_met' %(best)] = np.sum(np.asarray([[True if x>=init_params[-1] else False for x in c] \
                for c in np.asarray(pca_df[['probest_%i_%i'%(best,n) \
                                                   for n in range(best)]])]),axis=1)
    pca_df.insert(0,'covariance',value=bestmodel)
    pca_df.insert(1,'animal',value=mean_df.reset_index(False)['animal'])
    pca_df.insert(2,'cond',value=mean_df.reset_index(False)['cond'])
    pca_df.insert(3,'cell',value=mean_df.reset_index(False)['cell'])
    pca_df.insert(4,'week',value=mean_df.reset_index(False)['week'])
    pca_df.insert(5,'restraint',value=mean_df.reset_index(False)['restraint'])
    perc = np.round(100*(len(np.where(pca_df['%s_crit_met' %(best)]==1)[0])\
                         /len(pca_df['%s_crit_met' %(best)])),2)
    percents.append(perc)
    
    #extract mixture means for plotting based on individal components/features
    centers = gmm.means_
    
    #extract weight factors to show the covariance of cluster assignment
    w_factor = 0.2 / gmm.weights_.max()
    
    #plot the assignments (and label them)
    g=sns.scatterplot(x='P1', y='P2',hue=['clusters_%s' %(best)][0],edgecolor='k',\
                      palette=gmm_colors[:best],ax=ax,s=60,alpha=0.7,\
                          data=pca_df.loc[pca_df['%s_crit_met' %(best)]==1])
    g=sns.scatterplot(x='P1', y='P2',edgecolor='k',color=['yellow'],ax=ax,s=220,alpha=0.7, marker = '*',\
                          data=pca_df.loc[pca_df['%s_crit_met' %(best)]==0])
    ax.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=300, edgecolor='k')
    ellipseplot(gmm,ax,centers,gmm.covariances_,gmm_colors[:best])
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[x] for x in range(best)],[x+1 for x in range(best)],title='Cluster #',ncol=2)
    ax.set_title('%s, valid: %s' %(bestmodel.title(),np.round(perc,2)),fontsize=12,fontweight='bold')
    
    # store the gmm results
    gmm_outs.append(pca_df)
    gmm_centers.append([centers,w_factor])
    
plt.suptitle('Optimal Cluster Detection \n Bayesian Information Criterion (BIC)\nPCA components by cluster',\
             fontsize=20,fontweight='bold',y=1.0)
plt.savefig('GMM_N_%i_metric_flagged.pdf' %(len(pca_df)), bbox_inches="tight")
plt.close(fig)
            
fig,axes = plt.subplots(figsize=(15,15),nrows=2,ncols=2,sharey=True,sharex=True)
percents=[]
for ax,gmm_df,centers,gset,gmm in zip(axes.flatten(),gmm_outs,gmm_centers,gmm_set,gmms):
    best = gset[1]
    bestmodel = gset[0]
    centers = centers[0]
    
    #remove cells that do not meet criteria
    cleaned_gmm = gmm_df.loc[(gmm_df['%s_crit_met' %(best)]==1)].reset_index()
    
    g=sns.scatterplot(x='P1', y='P2',hue=['clusters_%s' %(best)][0],edgecolor='k',\
                      palette=gmm_colors[:best],ax=ax,s=60,alpha=0.7,data=cleaned_gmm)
    ax.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=300, edgecolor='k')
    ellipseplot(gmm,ax,centers,gmm.covariances_,gmm_colors[:best])
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[x] for x in range(best)],[x+1 for x in range(best)],title='Cluster #',ncol=2)
    ax.set_title('%s: %s' %(bestmodel.title(),np.round(100*(len(cleaned_gmm)/len(gmm_df)),2)),fontsize=12,fontweight='bold')
    percents.append(np.round(100*(len(cleaned_gmm)/len(gmm_df)),2))
plt.suptitle('Optimal Cluster Detection \n Bayesian Information Criterion (BIC)\nPCA components by cluster (CLEANED)' ,\
             fontsize=20,fontweight='bold',y=1)
plt.savefig('GMM_allcovariances.pdf' , bbox_inches="tight")
plt.close(fig)

#ask user what type of criteria to use for GMM component (cluster) numbers
message ='Select which covariance model you will be using from the following BIC results'+\
            ' where the botton details the shape, number of clusters, and percent of classified cells'+\
            ' using the confidence level set at: %s ' %(init_params[-1])
buttonset = [[x[0].title(),str(x[1]),str(y)] for x, y in zip(gmm_set,percents)]
buttonset.append('Choose own clsuter number\n(not advised)')
output = easygui.buttonbox(message, 'GMM metric selection',[str(x) for x in buttonset])

# parse out user response
if output =='Choose own clsuter number\n(not advised)':
    select = easygui.buttonbox('Choose number to components to cluster data into...',\
                             'GMM metric selection', [str(x) for x in range(2,8)])
    bic_ctype = easygui.buttonbox('Choose the covariance matrix shape to cluster data with...',\
                                 'GMM metric selection', ['spherical', 'tied', 'diag', 'full'])
    best =int(select)
    bestmodel = bic_ctype
else:
    import re
    best = int(output.split(',')[1].strip(' ').strip("'"))
    bestmodel = re.sub('\W', '', output.split(',')[0]).lower()

# Create directory that details the model and components
#create parent plot folder
os.chdir(gmm_folder)
if os.path.isdir(os.path.join(gmm_folder,'cov_raw_%s_comp_%s_%s' %(bestmodel,best,init_params[-1]))):
    shutil.rmtree(os.path.join(gmm_folder,'cov_raw_%s_comp_%s_%s' %(bestmodel,best,init_params[-1])))

os.mkdir('./cov_raw_%s_comp_%s_%s' %(bestmodel,best,init_params[-1]))
cov_folder = gmm_folder+'/cov_raw_%s_comp_%s_%s' %(bestmodel,best,init_params[-1])
os.chdir(cov_folder)

#create copy of PCs
pca_df = pd.DataFrame(X,columns=['P%i' %(x+1) for x in range(X.shape[1])])

#Run the GMM based on user input
gmm_colors=['lightseagreen','magenta','darkblue','orange','rebeccapurple','crimson']
gmm = mixture.GaussianMixture(n_components=best,
                covariance_type=bestmodel, random_state=0)
gmm.fit(pca_df)
gmms.append(gmm)

#add identifiers to dataframe
pca_df['clusters_%s' %(best)] = pd.to_numeric(gmm.predict(pca_df))

#calculate probability estimates of cluster assignment
weights = gmm.predict_proba(X)
for var,i in zip(['probest_%i_%i'%(best,x) for x in range(best)],range(best)):
    pca_df[var] = weights[:,i]
    
#create flag to denote cluster assignment that does not meet user's input
pca_df['%s_crit_met' %(best)] = np.sum(np.asarray([[True if x>=init_params[-1] else False for x in c] \
            for c in np.asarray(pca_df[['probest_%i_%i'%(best,n) \
                                               for n in range(best)]])]),axis=1)
pca_df.insert(0,'covariance',value=bestmodel)
pca_df.insert(1,'cond',value=mean_df.reset_index(False)['cond'])
pca_df.insert(2,'animal',value=mean_df.reset_index(False)['animal'])
pca_df.insert(3,'cell',value=mean_df.reset_index(False)['cell'])
pca_df.insert(4,'week',value=mean_df.reset_index(False)['week'])
pca_df.insert(5,'restraint',value=mean_df.reset_index(False)['restraint'])

#extract mixture means for plotting based on individal components/features
centers = gmm.means_

#extract weight factors to show the covariance of cluster assignment
w_factor = 0.2 / gmm.weights_.max()

#plot the assignments (and label them)
fig,ax = plt.subplots(figsize=(6,6))
g=sns.scatterplot(x='P1', y='P2',hue=['clusters_%s' %(best)][0],edgecolor='k',\
                  palette=gmm_colors[:best],ax=ax,s=60,alpha=0.7,\
                      data=pca_df.loc[pca_df['%s_crit_met' %(best)]==1])
g=sns.scatterplot(x='P1', y='P2',edgecolor='k',color=['yellow'],ax=ax,s=220,alpha=0.7, marker = '*',\
                      data=pca_df.loc[pca_df['%s_crit_met' %(best)]==0])
ax.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=300, edgecolor='k')
ellipseplot(gmm,ax,centers,gmm.covariances_,gmm_colors[:best])
h, l = ax.get_legend_handles_labels()
ax.legend([h[x] for x in range(best)],[x+1 for x in range(best)],title='Cluster #',ncol=2)
ax.set_title(bestmodel.title(),fontsize=12,fontweight='bold')
plt.savefig('GMM_N_%i_selected_flagged.pdf' %(len(pca_df)), bbox_inches="tight")
plt.close(fig)

#remove cells that do not meet criteria
fig,ax = plt.subplots(figsize=(6,6))
cleaned_gmm = pca_df.loc[(pca_df['%s_crit_met' %(best)]==1)].reset_index()

g=sns.scatterplot(x='P1', y='P2',hue=['clusters_%s' %(best)][0],edgecolor='k',\
                  palette=gmm_colors[:best],ax=ax,s=60,alpha=0.7,data=cleaned_gmm)
ax.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=300, edgecolor='k')
ellipseplot(gmm,ax,centers,gmm.covariances_,gmm_colors[:best])
h, l = ax.get_legend_handles_labels()
ax.legend([h[x] for x in range(best)],[x+1 for x in range(best)],title='Cluster #')
plt.suptitle('Optimal Cluster Detection \n Bayesian Information Criterion (BIC)\nPCA components by cluster (CLEANED)'+\
             '\n%s Covariance Matrix Shape' %(bestmodel.title()),\
             fontsize=20,fontweight='bold',y=1.1)
plt.savefig('GMM_N_%i_selected_cleaned.pdf' %(len(cleaned_gmm)), bbox_inches="tight")
plt.close(fig)

#create a pairplot
pcavars=['P%i' %(i+1) for i in range(X.shape[1])]
pcavars.append(['clusters_%s' %(best)][0])
g = sns.pairplot(cleaned_gmm[pcavars], hue=['clusters_%s' %(best)][0],\
             palette=gmm_colors[:best],plot_kws={'alpha':0.4})
handles = g._legend_data.values()
g._legend.remove()
g.fig.legend(handles=handles, labels=[x+1 for x in range(best)],\
             loc='center right', ncol=1,title='Cluster #')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Optimal Cluster Detection \n Bayesian Information Criterion (BIC)\nPCA Components',\
             fontsize=20,fontweight='bold',y=1.0)
g.savefig('GMM_N_%i_pairplot_PCAfeatures.pdf' %(len(cleaned_gmm)), bbox_inches="tight")
plt.close(g.fig)
pcavars.pop()

#create a pairplot by condition
pcavars=['P%i' %(i+1) for i in range(X.shape[1])]
for cond in cleaned_gmm.cond.unique():
    pcavars.append(['clusters_%s' %(best)][0])
    g = sns.pairplot(cleaned_gmm.loc[cleaned_gmm['cond']==cond][pcavars], hue=['clusters_%s' %(best)][0],\
                 palette=gmm_colors[:best],plot_kws={'alpha':0.4})
    handles = g._legend_data.values()
    g._legend.remove()
    g.fig.legend(handles=handles, labels=[x+1 for x in range(best)],\
                 loc='center right', ncol=1,title='Cluster #')
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Optimal Cluster Detection \n Bayesian Information Criterion (BIC)\nPCA Components %s' %(cond),\
                 fontsize=20,fontweight='bold',y=1.0)
    g.savefig('GMM_N_%i_pairplot_PCAfeatures_%s.pdf' %(len(cleaned_gmm.loc[cleaned_gmm['cond']==cond]),cond), bbox_inches="tight")
    plt.close(g.fig)
    pcavars.pop()

# Plot the PCA data as a function of condition assignment
pcavars=['P%i' %(i+1) for i in range(X.shape[1])]
pcavars.append('cond')
g = sns.pairplot(cleaned_gmm[pcavars], hue='cond',\
             palette=gmm_colors[:2],plot_kws={'alpha':0.4})
g.savefig('GMM_N_%i_pairplot_PCAfeatures_by_cond.pdf' %(len(cleaned_gmm)), bbox_inches="tight")
plt.close(g.fig)
pcavars.pop()

# Plot the PCA data as a function of condition assignment + Pre/Post 
cleaned_gmm['key'] = cleaned_gmm['cond']+'_'+cleaned_gmm['week'].astype(str)+'_'+cleaned_gmm['restraint'].astype(str)
pcavars=['P%i' %(i+1) for i in range(X.shape[1])]
pcavars.append('key')
g = sns.pairplot(cleaned_gmm[pcavars], hue='key',\
             palette='Set1',plot_kws={'alpha':0.4})
g.savefig('GMM_N_%i_pairplot_PCAfeatures_by_cond_recording.pdf' %(len(cleaned_gmm)), bbox_inches="tight")
plt.close(g.fig)
pcavars.pop()

#add oringinal variables back into the dataframe
merged_data= cleaned_gmm.merge(mean_df, on=['cond','animal', 'week','restraint','cell'])

#update identifiers
merged_data['cluster_num']  = merged_data['clusters_%s' %(best)]

#look at proportions of cells across weeks
cnts = merged_data.groupby(['cond','week', 'restraint','cluster_num'])['cell'].count().reset_index(name='cnt')
totals = merged_data.groupby(['cond','week', 'restraint'])['cell'].count().reset_index(name='cnt')

percent_df = []
for i,var in cnts.groupby(['cond','week', 'restraint','cluster_num']):
    var['percent'] = 100*(var['cnt'].iloc[0]/totals.loc[(totals['cond']==i[0])&(totals['week']==i[1])\
                        &(totals['restraint']==i[2])]['cnt'].iloc[0])
    percent_df.append(var)
percent_df = pd.concat(percent_df)

# Save outputs
merged_data.to_csv('GMM_out.csv')
percent_df.to_csv('GMM_count.csv')

#plot the percentage of cells by cluster and week, and condition
fig,axes = plt.subplots(figsize=(10,10),ncols=2,nrows=2,sharey=True,sharex=True)
for row,rest in zip(range(2),['pre','post']):
    for week,col in zip([1,4],range(2)):
        ax = axes[row,col]
        g = sns.barplot(data=percent_df.loc[(percent_df['week']==week)&\
                          (percent_df['restraint']==rest)],\
                    x="cluster_num", y='percent',\
                     hue="cond", hue_order=['CUS','EE'],
                     palette=['grey','mediumseagreen'],lw=3, \
                      errcolor="black",edgecolor="black", ax=ax)
        if rest =='post':
            ax.get_legend().set_visible(False)
            ax.set_xticklabels(range(1,best+1))
            ax.set_xlabel('Cluster Number',fontsize=14)
        else:
            ax.legend(title='Condition')
            ax.set_xticklabels('')
            ax.set_xlabel('')
        if col == 0:
            ax.set_ylabel('Percent of cells in cluster',fontsize=14)
        else:
            ax.set_ylabel('',fontsize=14)

        ax.set_title('Week: %i, %s Acute-Restraint' %(week,rest.title()), fontsize=14, fontweight='bold')
    
plt.suptitle('Condition impact on cluster assignment',fontsize=16)
fig.savefig('postGMM_cellpercentage.pdf' , bbox_inches="tight")
plt.close(fig)

# =============================================================================
# # Save the model to a file
# =============================================================================
import joblib
joblib.dump(gmm, 'gmm.joblib')

# =============================================================================
#        Plot pre and post values by cluster* condition* week
# =============================================================================
# initiate figure
fig,axes = plt.subplots(figsize=(20,20),nrows=3,ncols=7)
for cluster in sorted(merged_data.cluster_num.unique()):
    for var,label,cnt in zip(analysis_vars,['Peak Height', 'AUC', 'Width',\
                'Prominence','Rise Time', 'Decay Time', 'Number of Events'],\
                        range(len(analysis_vars))):
        # set axis
        ax = axes[int(cluster),cnt]
        
        # plot it
        g = sns.barplot(data=merged_data.loc[(merged_data['cluster_num']==cluster)\
                        &(merged_data['restraint']=='pre')], errorbar='se',\
                    x="week", y=var,hue="cond", hue_order=['CUS','EE'],
                     palette=[gmm_colors[int(cluster)],'white'],lw=3, \
                      errcolor="black",edgecolor="black", ax=ax)

        ax.set_ylabel('')
        ax.set_title(label,fontsize=14,fontweight='bold')
plt.suptitle('Condition impact on cluster assignment for pre-AR',fontsize=16)
fig.savefig('postGMM_pre_impact.pdf' , bbox_inches="tight")
plt.close(fig)

fig,axes = plt.subplots(figsize=(20,20),nrows=3,ncols=7)
for cluster in sorted(merged_data.cluster_num.unique()):
    for var,label,cnt in zip(analysis_vars,['Peak Height', 'AUC', 'Width',\
                'Prominence','Rise Time', 'Decay Time', 'Number of Events'],\
                        range(len(analysis_vars))):
        # set axis
        ax = axes[int(cluster),cnt]
        
        # plot it
        g = sns.barplot(data=merged_data.loc[(merged_data['cluster_num']==cluster)\
                        &(merged_data['restraint']=='post')], errorbar='se',\
                    x="week", y=var,hue="cond", hue_order=['CUS','EE'],
                     palette=[gmm_colors[int(cluster)],'white'],lw=3, \
                      errcolor="black",edgecolor="black", ax=ax)

        ax.set_ylabel('')
        ax.set_title(label,fontsize=14,fontweight='bold')
plt.suptitle('Condition impact on cluster assignment for post-AR',fontsize=16)
fig.savefig('postGMM_post_impact.pdf' , bbox_inches="tight")
plt.close(fig)

# =============================================================================
#                   Calculate the differences within cluster
# =============================================================================
# Calculate difference scores (condition-based)
diff_postGMM_df = []
for var in analysis_vars:
    # negative value = Post-acute restraint mean was higher
    for idx,df in merged_data.groupby(['cluster_num','cond','week']):
        
        # create x, y variables
        preset = df.loc[df['restraint']=='pre']
        postset = df.loc[df['restraint']=='post']
        
        # get sample size
        pre_n = len(preset); post_n = len(postset)
        
        # calculte pooled std and sems
        mean_diff = np.mean(postset[var])-np.mean(preset[var])
        std1 = np.std(preset[var],ddof=1); std2 = np.std(postset[var],ddof=1)
        pooled_std = np.sqrt(((len(preset[var])-1)*std1**2+(len(postset[var])-1)*std2**2)/\
                             (len(preset[var])+len(postset[var])-2))
        pooled_sem = pooled_std*np.sqrt((1/len(preset[var]))+(1/len(postset[var])))
        
        # run statistics
        t,p = stats.ttest_ind_from_stats(np.mean(preset[var]), std1, pre_n, np.mean(postset[var]), std2, post_n)
        out = pd.DataFrame(np.expand_dims(np.asarray([idx[0],idx[1],idx[2],var,pre_n,\
                                        post_n,mean_diff,std1,std2,pooled_std,pooled_sem,t,p]),1).T,\
            columns=['cluster_num','cond','week','variable','n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p'])
        out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']] = \
            out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']].astype(float) 
        diff_postGMM_df.append(out)
        
diff_postGMM_df = pd.concat(diff_postGMM_df)

# perform ttest on pooled means/stds (within condition)
diff_postGMM_df_week = diff_postGMM_df.copy()

diff_postGMM_df_week['tval']=''; diff_postGMM_df_week['pval']=''
for idx,df in diff_postGMM_df_week.groupby(['variable','cluster_num','cond']):
    df['week'] = df['week'].astype(float)
    
    pooled_1 = df.loc[df['week']==1]['n_pre']+df.loc[df['week']==1]['n_post']
    pooled_4 = df.loc[df['week']==4]['n_pre']+df.loc[df['week']==4]['n_post']
    t,p = stats.ttest_ind_from_stats(df.loc[df['week']==1]['mean'], df.loc[df['week']==1]['pooled_std'],\
                        pooled_1,df.loc[df['week']==4]['mean'], df.loc[df['week']==4]['pooled_std'], pooled_4)
    diff_postGMM_df_week.loc[(diff_postGMM_df_week['variable']==idx[0])&(diff_postGMM_df_week['cluster_num']==idx[1])&(diff_postGMM_df_week['cond']==idx[2]),'tval'] = t
    diff_postGMM_df_week.loc[(diff_postGMM_df_week['variable']==idx[0])&(diff_postGMM_df_week['cluster_num']==idx[1])&(diff_postGMM_df_week['cond']==idx[2]),'pval'] = p
diff_postGMM_df_week[['tval','pval']] = diff_postGMM_df_week[['tval','pval']].astype(float)
diff_postGMM_df_week.to_csv('week_differences_stats.csv')

# perform ttest on pooled means/stds
diff_postGMM_df['tval']=''; diff_postGMM_df['pval']=''
for idx,df in diff_postGMM_df.groupby(['variable','cluster_num','week']):

    pooled_cus = df.loc[df['cond']=='CUS']['n_pre']+df.loc[df['cond']=='CUS']['n_post']
    pooled_ee = df.loc[df['cond']=='EE']['n_pre']+df.loc[df['cond']=='EE']['n_post']
    t,p = stats.ttest_ind_from_stats(df.loc[df['cond']=='CUS']['mean'], df.loc[df['cond']=='CUS']['pooled_std'],\
                        pooled_cus,df.loc[df['cond']=='EE']['mean'], df.loc[df['cond']=='EE']['pooled_std'], pooled_ee)
    diff_postGMM_df.loc[(diff_postGMM_df['variable']==idx[0])&(diff_postGMM_df['cluster_num']==idx[1])&(diff_postGMM_df['week']==idx[2]),'tval'] = t
    diff_postGMM_df.loc[(diff_postGMM_df['variable']==idx[0])&(diff_postGMM_df['cluster_num']==idx[1])&(diff_postGMM_df['week']==idx[2]),'pval'] = p
diff_postGMM_df[['tval','pval']] = diff_postGMM_df[['tval','pval']].astype(float)

# create dataframe for condition-induced changes
diff_postGMM_df_cond = []

# initiate figure
fig,axes = plt.subplots(figsize=(14,14),nrows=3,ncols=7)
for cluster in sorted(diff_postGMM_df.cluster_num.unique()):
    for var,label,cnt in zip(analysis_vars,['Peak Height', 'AUC', 'Width',\
                'Prominence','Rise Time', 'Decay Time', 'Number of Events'],\
                        range(len(analysis_vars))):
        # set axis
        ax = axes[int(cluster),cnt]
        
        # plot it
        g = sns.barplot(data=diff_postGMM_df.loc[(diff_postGMM_df['variable']==var)\
                                &(diff_postGMM_df['cluster_num']==cluster)],ci=None,\
                    x="week", y='mean',hue="cond", hue_order=['CUS','EE'],
                     palette=[gmm_colors[int(cluster)],'white'],lw=3, \
                      errcolor="black",edgecolor="black", ax=ax)
        
        # Add error bars (from bootstrap)
        x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        sems = list(diff_postGMM_df.loc[(diff_postGMM_df['variable']==var)\
                                &(diff_postGMM_df['cluster_num']==cluster)]["pooled_sem"])
        if np.where(np.isnan(y_coords)==True)[0]:
            tmp = np.zeros_like(x_coords)
            for i in np.where(np.isnan(y_coords)==True):
                tmp[i] = np.nan
                sems.insert(i[0],np.nan)
        ax.errorbar(x=x_coords, y=y_coords, yerr=sems, fmt="none", c= "k",lw=2)
        ax.set_title(label,fontsize=14,fontweight='bold')
        
        # extract sig vals
        sigs = [float(x) for x in diff_postGMM_df.loc[(diff_postGMM_df['variable']==var)\
                      &(diff_postGMM_df['cond']=='CUS') &(diff_postGMM_df['cluster_num']==cluster)]['pval']]
        sigcolors = ['red' if x<=alpha else 'black' for x in sigs]
        
        # update y-label and x-labels and ticks
        ax.set_ylabel(''); ax.set_xlabel('')
        ax.set_xticklabels(['Week %i' %(x) for x in [1,4]],fontsize=14)
        [plt.setp(ax.get_xticklabels()[x], color=y) for x,y in zip(range(2),sigcolors) ]
        
        if var !='width':
            ax.get_legend().set_visible(False)
        else:
            ax.legend(title='Condition',bbox_to_anchor=(1, 0.75))
        if cnt in [0,3,6]:
            ax.set_ylabel('Change Score (post-pre)')
    
        for idx,df in diff_postGMM_df.loc[(diff_postGMM_df['variable']==var)&\
                                          (diff_postGMM_df['cluster_num']==cluster)].groupby('week'):
            
            # create x, y variables
            preset = df.loc[df['cond']=='CUS']
            postset = df.loc[df['cond']=='EE']
            
            # calculte pooled std and sems
            mean_diff = postset['mean']-preset['mean']
            std1 = preset['pooled_std']; std2 = postset['pooled_std']
            
            # get sample sizes
            pooled_ee = postset['n_pre']+postset['n_post']
            pooled_cus = preset['n_pre']+preset['n_post']
            
            pooled_std = np.sqrt(((pooled_cus-1)*std1**2+(pooled_ee-1)*std2**2)/\
                                 (pooled_cus+pooled_ee-2))
            pooled_sem = pooled_std*np.sqrt((1/pooled_cus)+(1/pooled_ee))
            
            # run statistics
            t,p = stats.ttest_ind_from_stats(preset['mean'], std1, pooled_cus, postset['mean'], std2, pooled_ee)
            
            # create dataframe
            out = pd.DataFrame(np.expand_dims(np.asarray([cluster,idx[0],var,pooled_cus,\
                                            pooled_ee,mean_diff,std1,std2,pooled_std,pooled_sem,t,p]),1).T,\
                columns=['cluster_num','week','variable','n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p'])
            out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']] = \
                out[['n_pre','n_post','mean','std1','std2','pooled_std','pooled_sem','t','p']].astype(float) 
            diff_postGMM_df_cond.append(out)

# Save the figure
plt.suptitle('Condition impact on calcium properties, raw', fontsize=15)
plt.savefig('grouped_raw_data_nopca.pdf', transparent=False, bbox_inches="tight")
plt.close()

# Concatenate frames
diff_postGMM_df_cond = pd.concat(diff_postGMM_df_cond)

# perform ttest on pooled means/stds
diff_postGMM_df_cond['tval']=''; diff_postGMM_df_cond['pval']=''
for idx,df in diff_postGMM_df_cond.groupby(['cluster_num','variable']):

    pooled_1 = df.loc[df['week']==str(1)]['n_pre']+df.loc[df['week']==str(1)]['n_post']
    pooled_4 = df.loc[df['week']==str(4)]['n_pre']+df.loc[df['week']==str(4)]['n_post']
    t,p = stats.ttest_ind_from_stats(df.loc[df['week']==str(1)]['mean'], df.loc[df['week']==str(1)]['pooled_std'],\
                        pooled_1,df.loc[df['week']==str(4)]['mean'], df.loc[df['week']==str(4)]['pooled_std'], pooled_4)
    diff_postGMM_df_cond.loc[(diff_postGMM_df_cond['cluster_num']==idx[0])&\
                            (diff_postGMM_df_cond['variable']==idx[1]),'tval'] = t
    diff_postGMM_df_cond.loc[(diff_postGMM_df_cond['cluster_num']==idx[0])&\
                            (diff_postGMM_df_cond['variable']==idx[1]),'pval'] = p
diff_postGMM_df_cond[['tval','pval']] = diff_postGMM_df_cond[['tval','pval']].astype(float)

# create x-positions
diff_postGMM_df_cond['xpos'] = ''
for cluster in sorted(diff_postGMM_df.cluster_num.unique()):
    for x,var in zip(range(len(analysis_vars)),analysis_vars):
        diff_postGMM_df_cond.loc[(diff_postGMM_df_cond['cluster_num']==cluster)&\
                                 (diff_postGMM_df_cond['variable']==var)&\
                                (diff_postGMM_df_cond['week']==str(1)),'xpos'] = x-0.15
        diff_postGMM_df_cond.loc[(diff_postGMM_df_cond['cluster_num']==cluster)&\
                                 (diff_postGMM_df_cond['variable']==var)&\
                                (diff_postGMM_df_cond['week']==str(4)),'xpos'] = x+0.15

cond_sigs = ['orange' if x<0.05 else 'black' for x in np.asarray(diff_postGMM_df_cond['p'])]
sigs = ['red' if x<0.05 else 'black' for x in np.asarray(diff_postGMM_df_cond.loc[diff_postGMM_df_cond['week']==str(1)]['pval'])]

# plot it (all data)
fig, axes = plt.subplots(figsize=(10, 10),ncols=7,nrows=len(diff_postGMM_df_cond.cluster_num.unique()))
for cluster,color in zip(sorted(diff_postGMM_df_cond.cluster_num.unique()),\
                         gmm_colors[:len(diff_postGMM_df_cond.cluster_num.unique())]):
    for var,idx,label in zip(analysis_vars,range(len(analysis_vars)),\
            ['Peak Height', 'AUC', 'Width', 'Prominence',\
             'Rise Time', 'Decay Time', 'Number of Events']):
        dset = diff_postGMM_df_cond.loc[(diff_postGMM_df_cond['cluster_num']==cluster)&\
                             (diff_postGMM_df_cond['variable']==var)]
        axes[int(cluster),idx].errorbar(range(2),dset['mean'],lw=3,alpha=0.6,\
                    yerr = dset['pooled_sem'], ls = "None",ecolor=color)
        axes[int(cluster),idx].scatter(range(2),dset['mean'],\
                   s = 100, marker = "h",color=color)
        axes[int(cluster),idx].set_xlim(-1,2)
        axes[int(cluster),idx].set_title(label,fontsize=14,fontweight='bold')
        axes[int(cluster),idx].set_xticks([0,1]); 
        axes[int(cluster),idx].set_xticklabels(['Week 1', 'Week 4'],fontsize=13)
        axes[int(cluster),idx].axhline(0)

# Save the figure
plt.suptitle('Condition impact on calcium properties,\nchange score (negative: CUS stronger AR difference', fontsize=15)
plt.savefig('grouped_changescore_data_v1.pdf', transparent=False, bbox_inches="tight")
plt.close()

# =============================================================================
#                    Create waveforms from data
# =============================================================================
# The formula for this comes from this paper (CITE)
waveform_df = merged_data.groupby(['cluster_num','week','restraint','cond'])[analysis_vars].mean().reset_index()
waveform_df['time'] = ''; waveform_df['wave'] = ''
for idx,df in waveform_df.iterrows():
    
    trace_length = 5000     #in milisceconds
    peak_height,event_width,rise_t,decay_time,area,event_count = \
        df[['filtsig','width','rise_t', 'decay_t','auc', 'peak_total']]
        
    t = np.linspace(0, 12, 1000)
    signal = peak_height * np.exp(-t / rise_t) * (1-np.exp(-t /decay_time))

    signal *= area / np.sum(signal)

    # Store it
    waveform_df.iloc[idx,-2] = np.expand_dims(t,1)
    waveform_df.iloc[idx,-1] = np.expand_dims(signal,1)

# Save waveform dataframe
waveform_df.to_csv('waveform_df.csv')

# Plot the signals based on condition
fig,axes= plt.subplots(figsize=(8,8),ncols=2,nrows=2,sharey=True,sharex=True)
axes_list = axes.flatten()
for idx,df in waveform_df.groupby(['week','cond']):

    if idx[1]=='CUS':
        row = 0
    else:
        row = 1
    if idx[0] == 1:
        col = 0
    else:
        col = 1
    
    inset = axes[row,col].inset_axes([0.6,0.6,0.35,0.35])
    g = sns.barplot(data = df, x = 'restraint',y='peak_total',\
                    order=['pre','post'],hue='cluster_num',\
                    hue_order=sorted(df.cluster_num.unique()),
                    palette=gmm_colors,ax=inset)
    inset.legend().set_visible(False)
    inset.set_xticklabels(['Pre','Post'],fontsize=8)
    inset.set_xlabel('Restraint',fontsize=9)
    inset.set_ylabel('Total events',fontsize=9)
    inset.set_ylim(0,120)
    inset.tick_params(axis='both', which='major', labelsize=8)
    
    # create dotted bars, in post, to match the waveforms
    for i, bar in enumerate(inset.patches):
        bar.set_edgecolor(np.repeat(gmm_colors,2)[i])
        if i % 2 != 0:
            bar.set_hatch('..')
            bar.set_facecolor('white')
            #bar.set_facecolor(np.repeat(gmm_colors,2)[i])
            
    for cluster in sorted(df.cluster_num.unique()):
        for restraint, lstyle in zip(['pre','post'],['-',':']):
            dset = df.loc[(df['restraint']==restraint)&(df['cluster_num']==cluster)]
            axes[row,col].plot(np.asarray(dset['time'].iloc[0]).T,\
                                  np.asarray(dset['wave'].iloc[0]).T,\
                color=gmm_colors[int(dset['cluster_num'])],ls=lstyle)
    axes[row,col].set_title('Week: %i, %s' %(idx[0],idx[1]))
    
    # Formatting
    axes[0,0].set_ylabel('Normalized\n'+r'Fluorescence ($\Delta$F/F)',fontsize=13)
    axes[1,0].set_ylabel('Normalized\n'+r'Fluorescence ($\Delta$F/F)',fontsize=13)
    axes[1,0].set_xlabel('Time (sec)',fontsize=13)
    axes[1,1].set_xlabel('Time (sec)',fontsize=13)
    
# Save the figure
plt.suptitle('Condition impact on calcium waveforms', fontsize=15)
plt.savefig('waveform_dynamics.pdf', transparent=False, bbox_inches="tight")
plt.close()

# Normalize the data
properties = ['filtsig', 'auc', 'width','prominence', 'rise_t', 'decay_t', 'peak_total']
waveform_z_df = waveform_df.copy()
waveform_z_df[properties] = (waveform_z_df[properties] - waveform_z_df[properties].mean()) / waveform_z_df[properties].std()

# Create the data and use this for mins/maxs
radar_data = []; radar_vals = []
for cond in waveform_z_df.cond.unique():
    for cluster in sorted(waveform_z_df.cluster_num.unique()):
        dataset =[]
        for week in sorted(waveform_z_df.week.unique()):
            dset = waveform_z_df.loc[(waveform_z_df['cond']==cond)&(waveform_z_df['cluster_num']==cluster)\
                                    &(waveform_z_df['week']==week)]
            change_df = dset.loc[dset['restraint']=='post'].reset_index(drop=True)
            
            # normalize the values before subtracting
            change_df[properties]=change_df[properties]-dset.loc[dset['restraint']=='pre'].reset_index(drop=True)[properties]
            dataset.append((week,list(change_df[properties].iloc[0]))); radar_vals.append(list(change_df[properties].iloc[0]))

# Get mins and maxs to scale the figures appropriately
radar_min = min(np.concatenate([x for x in radar_vals]))
radar_max = max(np.concatenate([x for x in radar_vals]))

fig,axes = plt.subplots(figsize=(10,10),subplot_kw={'projection': 'polar'},\
                        ncols=len(waveform_z_df.cluster_num.unique()),nrows=2)
axes_list = axes.flatten(); cnt=0
for cond in waveform_z_df.cond.unique():
    for cluster in sorted(waveform_z_df.cluster_num.unique()):
        dataset =[]
        for week in sorted(waveform_z_df.week.unique()):
            dset = waveform_z_df.loc[(waveform_z_df['cond']==cond)&(waveform_z_df['cluster_num']==cluster)&(waveform_z_df['week']==week)]
            change_df = dset.loc[dset['restraint']=='post'].reset_index(drop=True)
            
            # normalize the values before subtracting
            change_df[properties]=change_df[properties]-dset.loc[dset['restraint']=='pre'].reset_index(drop=True)[properties]
            dataset.append((week,list(change_df[properties].iloc[0]))); radar_vals.append(list(change_df[properties].iloc[0]))
        
        # Plot it
        create_radar_plot(dataset, properties, axes_list[cnt],ylim=(np.round(radar_min*1.1,2),\
                       np.round(radar_max*1.1,2)),ypad=20,title='%s: %s' %(cond,cluster+1),colors=['gold','slateblue'])
        axes_list[cnt].set_yticklabels([])
        

        cnt+=1
fig.set_size_inches(18.5,10.5)
plt.savefig('radar_plots.pdf', transparent=False, bbox_inches="tight")
plt.close()

# =============================================================================
# =============================================================================
#       LOAD FULL TRACE FILE TO PERFORM ENSEMBLE SYNCHRONY FUNCTIONS
# =============================================================================
# =============================================================================
os.chdir(dir_folder)
file_list = os.listdir('./')

#locates the property files (will use as identity flagging set)
trace_files = easygui.multchoicebox(
        msg = 'Select the files you want to use for this analysis...', 
        choices = (sorted([x if 'traces' in x else 'No other options found'for x in file_list ],reverse=True))) 

trace_df = []
for df in tqdm(trace_files,position=0, leave=True, colour='green',\
                       desc = 'Creating the trace dataframe'):
    tmp = pd.read_csv(df,sep=",", low_memory=False)
    if 'CUS' in df:
        cond = 'CUS'

    else:
        cond = 'EE'

    tmp.insert(0,'cond',cond)
    trace_df.append(tmp)
    
trace_df = pd.concat(trace_df)

# Batch process valid cell extracttion
os.chdir(cov_folder)
file_list = os.listdir('./')
if 'valid_traces.pickle' not in file_list:
    valid_df = applyParallel_trace(merged_data.groupby(['cond','animal', 'week', 'restraint']), clipping_par)
    valid_df = pd.concat(valid_df)
    
    # Save output so this does not have to be done again
    valid_df.to_pickle('valid_traces.pickle')
    
else:
    # load traces
    valid_df = pd.read_pickle('valid_traces.pickle')

valid_df.groupby(['cond','animal','order'])['order'].nunique()

# =============================================================================
#       ASK USER TO CHOOSE ANIMAL, CONDITION, CELLS to plot (over eachother)
# =============================================================================
import copy
def normalizedata(data, setrange=(0,1)):
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm * (setrange[1] - setrange[0]) + setrange[0]

plot_cond = easygui.buttonbox(msg = 'Select the condition you want to plot...', 
        choices = (sorted([x for x in  valid_df.cond.unique()]))) 
plot_animal = easygui.buttonbox(msg = 'Select the animal you want to plot...', 
        choices = (sorted([x for x in  valid_df.loc[valid_df.cond == plot_cond].animal.unique()]))) 
plot_restraint = easygui.buttonbox(msg = 'Select the restraint you want to plot...', 
        choices = (sorted([str(x) for x in valid_df.loc[(valid_df.cond == plot_cond)&(valid_df.animal == plot_animal)].order.unique()]))) 
plot_restraint = int(plot_restraint)

# query the data
valid_query = valid_df.loc[(valid_df.cond == plot_cond)&(valid_df.animal == plot_animal)\
                           &(valid_df.order == plot_restraint)]
    
# Allow user to select cells to plot
cell_select = easygui.multchoicebox(
        msg = 'Select the cells you want to plot for this visualization...', 
        choices = (sorted([x for x in valid_query.cell.unique()])))

# Allow user create the order
cell_order =[]; cell_temp = copy.deepcopy(cell_select); cell_colors = []
for i in range(len(cell_select)-1):
    c_num = easygui.buttonbox(
            msg = 'Select the cell you want to plot in position %s...' %(i+1), 
            choices = (sorted([x for x in cell_temp])))
    cell_temp.remove(c_num); cell_order.append(c_num)
    cell_color = easygui.buttonbox(msg = 'Select the color for cell: %s' %(c_num), 
            choices = ([x for x in gmm_colors]))
    cell_colors.append(cell_color)
cell_color = easygui.buttonbox(msg = 'Select the color for cell: %s' %(cell_temp[0]), 
        choices = ([x for x in gmm_colors]))
cell_colors.append(cell_color)
cell_order.append(cell_temp[0]); 
cell_ylabels = copy.deepcopy(cell_order[::-1])
cell_ylabels.insert(0,0)

# Set the plotting time
t_sec = 600     #time in seconds

# Create the figure (in black)
fig,ax = plt.subplots(figsize=(15,8))
for cell,idx in zip(cell_order[::-1],range(len(cell_order))):
    plot_t = valid_query.loc[valid_query['time_sec'].between(0,t_sec)]['time_sec'].unique()
    plot_trace = valid_query.loc[valid_query['cell']==cell]['filtsig'][:len(plot_t)]
    ax.plot(plot_t,normalizedata(plot_trace,setrange=(0,0.8))+(idx+1),lw=2,color='black')
ax.set_yticklabels(cell_ylabels)
ax.set_ylabel('Cell Number',fontweight='bold',fontsize=16)
ax.set_xlabel('Time from recording (sec)',fontweight='bold',fontsize=16)
ax.set_title('%s: %s, Recording #: %s' %(plot_animal.title(),plot_cond,plot_restraint),fontsize=18)
plt.savefig('representative_traces_black.pdf', transparent=True, bbox_inches="tight")
plt.close()

# Create the figure (in red)
fig,ax = plt.subplots(figsize=(15,8))
for cell,idx in zip(cell_order[::-1],range(len(cell_order))):
    plot_t = valid_query.loc[valid_query['time_sec'].between(0,t_sec)]['time_sec'].unique()
    plot_trace = valid_query.loc[valid_query['cell']==cell]['filtsig'][:len(plot_t)]
    ax.plot(plot_t,normalizedata(plot_trace,setrange=(0,0.8))+(idx+1),lw=2,color='crimson')
ax.set_yticklabels(cell_ylabels)
ax.set_ylabel('Cell Number',fontweight='bold',fontsize=16)
ax.set_xlabel('Time from recording (sec)',fontweight='bold',fontsize=16)
ax.set_title('%s: %s, Recording #: %s' %(plot_animal.title(),plot_cond,plot_restraint),fontsize=18)
plt.savefig('representative_traces_red.pdf', transparent=True, bbox_inches="tight")
plt.close()

# Create the figure (colored by cluster number)
fig,ax = plt.subplots(figsize=(15,8))
for cell,idx in zip(cell_order[::-1],range(len(cell_order))):
    plot_t = valid_query.loc[valid_query['time_sec'].between(0,t_sec)]['time_sec'].unique()
    plot_trace = valid_query.loc[valid_query['cell']==cell]['filtsig'][:len(plot_t)]
    ax.plot(plot_t,normalizedata(plot_trace,setrange=(0,0.8))+(idx+1),lw=2,color=cell_colors[::-1][idx])
ax.set_yticklabels(cell_ylabels)
ax.set_ylabel('Cell Number',fontweight='bold',fontsize=16)
ax.set_xlabel('Time from recording (sec)',fontweight='bold',fontsize=16)
ax.set_title('%s: %s, Recording #: %s' %(plot_animal.title(),plot_cond,plot_restraint),fontsize=18)
plt.savefig('representative_traces_colored.pdf', transparent=True, bbox_inches="tight")
plt.close()

# =============================================================================
#                         FLIP THROUGH EACH PEAK TO GET:
#                          1) inter-event interval (IEI)
#                          2) resting fluorescence (rf)
#
#       Based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5553047/
#
#           The baseline calcium fluorescence of a neuron is an indication 
#       of the intracellular free calcium concentration, [Ca2+]i, and is related 
#       to resting membrane potential, Vm
# =============================================================================
# Create directory to store the synchrony outputs
os.chdir(cov_folder)
if os.path.isdir(os.path.join(cov_folder,'synchrony')):
    shutil.rmtree(os.path.join(cov_folder,'synchrony' ))

os.mkdir('./synchrony')
synch_folder = cov_folder+'/synchrony' 
os.chdir(synch_folder)

# Set the sampling rate
srate = 20

# Check for a dataframe that already has the instantaneous phases caclulated
# If one is found, user will be asked if they want to use that one, or to create 
# a new one
message = 'A dataframe has been been found with the instantaneous phase having'\
        + ' already been calculated. Would you like to use this one, or create'\
        + ' and save another one?'        
buttonset = ['Load one', 'Create new one']
load=False

if len([x for x in file_list if 'instantaneous_phase_postGMM' in x])>0:
    output = easygui.buttonbox(message, 'Instantaneous phase calculation', buttonset)
    
    if output == 'Load one':
        # Set variable
        load = True
        
        # Provide a user box for analyst to choose from
        pickles = easygui.multchoicebox(
                msg = 'Select the files you want to load for this analysis...', 
                choices = (sorted([x if 'instantaneous' in x else 'No other options found'for x in os.listdir('./')],reverse=True)))
    
        # Load file
        out_df = pd.read_pickle(pickles[0])
        
    if output != 'Load one':
        load = False
        
if load == False:
        
        # Batch process the cell-classifier function (which also computes instantaneous phase)
        out_df = applyParallel(valid_df.groupby(['animal','fname','cell','order']), classifier_par)
        
        # Save output (in case memory is overloaded)
        out_df.to_pickle('instantaneous_phase_postGMM_%s.pickle' %(time.strftime("%Y%m%d")))

# Create pair dataframe that is agnostic to cluster assignment
pair_agnostic_df = []
for idx,var in tqdm(out_df.groupby(['cond','animal']),position=0, leave=True, colour='green',\
                       desc = 'Creating dataframe for synchrony calculations'):
    for order in sorted(var['order'].unique()):
        pos = order
        if order ==7:
            pos = 3
        if order ==8:
            pos =4
        
        dset = var.loc[(var['order']==order)]           
        applied = valid_df.loc[(valid_df['cond']==var['cond'].iloc[0])&\
                               (valid_df['animal']==var['animal'].iloc[0])&(valid_df['order']==order)]
        dset.insert(5,'applied_cluster',value = '')
        for cell in dset.cell.unique():
            dset.loc[(dset.cell==cell), 'applied_cluster'] =\
                int(applied.loc[(applied.cell==cell)]['applied_cluster'].iloc[0]+1)
        dset['applied_cluster'] = dset['applied_cluster'].astype(int) 
        
        cell_pairs = list(permutations(dset['cell'].unique(), 2))
        #cell_pairs = list(combinations(dset['cell'].unique(), 2))
        for cell1,cell2 in cell_pairs:
            cell1_clust = dset.loc[dset.cell==cell1]['applied_cluster'].iloc[0]
            cell2_clust = dset.loc[dset.cell==cell2]['applied_cluster'].iloc[0]
            df_out = pd.DataFrame(np.expand_dims(np.asarray([idx[0],idx[1],order,cell1_clust,cell2_clust,'intra',cell1,cell2]),0),\
                                  columns=['cond','animal','order','applied_cluster','compared_cluster','type','cell1','cell2'])
            df_out[['cell1_phase','cell2_phase']] = [[dset.loc[dset['cell']==cell1]['instant_phase'].iloc[0],\
                                                            dset.loc[dset['cell']==cell2]['instant_phase'].iloc[0]]]
            
            # ensure lengths are the same (sometimes off by 1-2 samples)
            min_len = min(min([[dset.loc[dset['cell']==cell1]['instant_phase'].iloc[0].shape[0],\
                            dset.loc[dset['cell']==cell2]['instant_phase'].iloc[0].shape[0]]]))
            df_out['phase_diff'] = [np.mod(abs(df_out['cell1_phase'].iloc[0][:min_len]-\
                                    df_out['cell2_phase'].iloc[0][:min_len]),2*np.pi)]
            pair_agnostic_df.append(df_out)

pair_agnostic_df = pd.concat(pair_agnostic_df)

# Parallelize the phase-synchrony function
phase_agnostic_synch_df = applyParallel_synch(pair_agnostic_df.groupby(['cond','animal', 'order', 'applied_cluster']), phase_synch_par)
phase_agnostic_synch_df.to_pickle('phase_synch_agnostic_postGMM_%s.pickle' %(time.strftime("%Y%m%d")))

# Query out high synchrony cells to plot as representative traces
high_synch = phase_agnostic_synch_df.loc[(phase_agnostic_synch_df.phase_synch>=0.65)]

def traceplot(row,traces):
    query1 = traces.loc[(traces.animal==row['animal'].iloc[0])&\
        (traces.order==int(row['order'].iloc[0]))& (traces.cell==row['cell1'].iloc[0])]
    query2 = traces.loc[(traces.animal==row['animal'].iloc[0])&\
        (traces.order==int(row['order'].iloc[0]))& (traces.cell==row['cell2'].iloc[0])]
    
        
    fig,ax = plt.subplots(figsize=(15,8))    
    for cname, cvals, color in zip([query1['cell'].iloc[0],query2['cell'].iloc[0]],\
             [query1.filtsig,query2.filtsig],['darkblue','mediumseagreen']):
        ax.plot(query1.time_sec,cvals,label=cname,color=color,alpha=0.6,lw=3)
    ax.set_ylabel(r'Fluorescence ($\Delta$F/F)',fontsize=13)
    ax.set_xlabel('Time from recording start (seconds)',fontsize=15)
    ax.legend(title='Cell#')
    ax.set_title('Representative cell trace for phase synchrony'+\
                 '\nAnimal: %s, Recording: %s' %(row['animal'].iloc[0].title(),\
                                                 row['order'].iloc[0])+\
                   '\nPhase Synchrony = %s' %(np.round(row.phase_synch.iloc[0],3)),\
                    fontsize=17,fontweight='bold')
    plt.savefig('rep_%s_%s_rec%s_%s_%s.pdf' %(row['cond'].iloc[0],row['animal'].iloc[0].title(),\
            row['order'].iloc[0],row['cell1'].iloc[0],row['cell2'].iloc[0]), bbox_inches="tight")
    plt.close(fig)
        
for x in high_synch.groupby(['cond','animal','order','applied_cluster','compared_cluster','type']):
    try:
        traceplot(x[1],valid_df)
    except:
        pass

# Query out low synchrony cells to plot as representative traces
low_synch = phase_agnostic_synch_df.loc[(phase_agnostic_synch_df.phase_synch<=0.15)]

for x in low_synch.groupby(['cond','animal','order','applied_cluster','compared_cluster','type'])[:15]:
    try:
        traceplot(x[1],valid_df)
    except:
        pass

# For each recording session, look at heatmap of synchrony
#Set normalize parameter
normval = 'yes'             

# For each recording session, look at heatmap of synchrony
synch_value_df =[]; synch_norm_df =[]
for idx,df in phase_agnostic_synch_df.groupby(['cond','animal']):
  
    # Get max synch values
    max_val = np.round(max(df['phase_synch']),2)
    
    # Calculate mean values
    df.loc[df['applied_cluster']!=df['compared_cluster'],'type'] = 'inter'
    
    mean_synch = df.groupby(['cond','animal','order',\
                  'applied_cluster','type'])['phase_synch'].mean().reset_index()
    synch_value_df.append(mean_synch)
    
    # Initiate figure and colorbar
    fig,axes = plt.subplots(figsize=(12,12),nrows=2,ncols=2)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    axes_list = axes.flatten()
    
    for order_idx,var in df.groupby('order'):
        if order_idx == str(7):
            order_idx = 3
        if order_idx == str(8):
            order_idx = 4
        
        ax = axes_list[int(order_idx)-1]
        
        if normval == 'yes':
            var['phase_synch']=(var['phase_synch']-\
                    var['phase_synch'].min())/(var['phase_synch'].max()-var['phase_synch'].min())
            mean_synch = var.groupby(['cond','animal','order',\
                          'applied_cluster','type'])['phase_synch'].mean().reset_index()
            synch_norm_df.append(mean_synch)
                
            max_val = 1
        
        # pivot the data
        pivotset = pd.pivot(var,index=['cell1','applied_cluster'],\
                            columns=['compared_cluster','cell2'],values='phase_synch')
        
        # sort the data to show unity line
        pivotset = pivotset.sort_index(level=['applied_cluster','cell1'],ascending=[True,True])
        pivotset.sort_index(axis=1, level=[0, 1], ascending=[True, True], inplace=True)
            
        # count number of cells in each cluster
        clust_cnt = var.groupby(['applied_cluster'])['cell1'].nunique().reset_index()
        
        x_starts = list(np.cumsum(clust_cnt['cell1'])-clust_cnt['cell1'])
        x_ends = list(np.cumsum(clust_cnt['cell1']))
        x_starts.insert(len(x_starts),0); x_ends.insert(0,0)
        
    
        # Generate heatmap w/ boxes around intra-cluster comparisons
        sns.heatmap(pivotset, annot=False, cmap='coolwarm',ax=ax,\
                        vmin=0,vmax=max_val,\
                        cbar_ax=cbar_ax)
 
        for i in range(len(clust_cnt)):
            ax.plot([x_starts[i]+0.1,x_starts[i]+0.1],[x_ends[i],x_ends[i+1]],lw=3,color='black') #Left
            if i == 0:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i]+(0.1*i),x_ends[i]+(0.1*i)],lw=3,color='black') #Top
            else:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i],x_ends[i]],lw=3,color='black') #Top
            
            if i == len(clust_cnt)-1:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i+1]-(0.1*i),x_ends[i+1]-(0.1*i)],lw=3,color='black') #Bottom
            else:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i+1],x_ends[i+1]],lw=3,color='black') #Bottom
            
            if i < len(clust_cnt)-1:
                ax.plot([x_ends[i+1]+0.1,x_ends[i+1]+0.1],[x_ends[i],x_ends[i+1]],lw=3,color='black') #Right
            else:
                ax.plot([x_ends[i+1]-(0.1*i),x_ends[i+1]-(0.1*i)],[x_ends[i],x_ends[i+1]],lw=3,color='black') #Right

    # Save the figure
    plt.suptitle('%s: %s - Phase synchrony' %(df['cond'].iloc[0],df['animal'].iloc[0]))
    fig.savefig('%s_%s_phase_synch.pdf' %(df['cond'].iloc[0],df['animal'].iloc[0]),bbox_inches = 'tight')
    plt.close(fig)

# Concatenate mean dataframes
synch_value_df = pd.concat(synch_value_df)
synch_norm_df = pd.concat(synch_norm_df)

# Query out datasets that have intra and inter for a given session
cnt_set = synch_value_df.groupby(['cond','animal','order',\
                    'applied_cluster'])['type'].count().reset_index(name='cnt')
valid_synch = []; norm_synch =[]
for idx,row in cnt_set.iterrows():
    if row['cnt']==2:
        valid_synch.append(synch_value_df.loc[(synch_value_df['cond']==row['cond'])&\
                  (synch_value_df['animal']==row['animal'])&\
                (synch_value_df['order']==row['order'])&\
                    (synch_value_df['applied_cluster']==row['applied_cluster'])])
        norm_synch.append(synch_norm_df.loc[(synch_norm_df['cond']==row['cond'])&\
                  (synch_norm_df['animal']==row['animal'])&\
                (synch_norm_df['order']==row['order'])&\
                    (synch_norm_df['applied_cluster']==row['applied_cluster'])])
valid_synch = pd.concat(valid_synch)
norm_synch = pd.concat(norm_synch)

# Update orders
valid_synch.loc[valid_synch['order']==str(7),'order']=3
valid_synch.loc[valid_synch['order']==str(8),'order']=4
norm_synch.loc[norm_synch['order']==str(7),'order']=3
norm_synch.loc[norm_synch['order']==str(8),'order']=4

# divide each intra by inter to compare across sessions*animals
norm_df=[]; 
for idx,df in valid_synch.groupby(['cond','animal','order','applied_cluster']):
    tmp = df.loc[df['type']=='intra'].reset_index()
    tmp['norm'] = tmp['phase_synch']/np.mean(df.loc[df['type']=='inter']['phase_synch'])
    norm_df.append(tmp)

norm_df_2= []
for idx,df in norm_synch.groupby(['cond','animal','order','applied_cluster']):
    tmp = df.loc[df['type']=='intra'].reset_index()
    tmp['norm'] = tmp['phase_synch']/np.mean(df.loc[df['type']=='inter']['phase_synch'])
    norm_df_2.append(tmp)

norm_df = pd.concat(norm_df)

# Plot results
norm_df['week']=1; norm_df.loc[norm_df['order'].isin([3,4]),'week']=4
fig,axes = plt.subplots(figsize=(8,5),ncols=2,sharey=True,sharex=True)
for cond,ax in zip(['CUS','EE'],axes.flatten()):
    g = sns.barplot(x='applied_cluster',y='norm',hue='week',\
            order=[str(1),str(2),str(3)],ax=ax,\
            data=norm_df.loc[norm_df['cond']==cond],errorbar=('ci',68),
            palette=['slategray','crimson'],lw=3,\
                errcolor='black',edgecolor='black')
    ax.axhline(1,ls=':',color='black',lw=2)
    ax.set_title(cond,fontsize=16,fontweight='bold')
    if cond=='CUS':
        ax.set_ylabel('Normalized synchrony',fontsize=16)
    else:
        ax.set_ylabel('')
fig.savefig('phase_synch_normalized.pdf' ,bbox_inches = 'tight')
plt.close(fig)

fig,axes = plt.subplots(figsize=(5,10),nrows=2,sharey=True,sharex=True)
for cond,ax in zip(['CUS','EE'],axes.flatten()):
    g = sns.barplot(x='applied_cluster',y='norm',hue='week',\
            order=[str(1),str(2),str(3)],ax=ax,\
            data=norm_df.loc[norm_df['cond']==cond],errorbar=('ci',68),
            palette=['slategray','crimson'],lw=3,\
                errcolor='black',edgecolor='black')
    ax.axhline(1,ls=':',color='black',lw=2)
    ax.set_title(cond,fontsize=16,fontweight='bold')
    ax.set_ylabel('Normalized synchrony',fontsize=16)

fig.savefig('phase_synch_normalized2.pdf' ,bbox_inches = 'tight')
plt.close(fig)


sns.catplot(x='applied_cluster',y='phase_synch',hue='type',col='cond',\
            row='order',data=valid_synch,kind='bar',errorbar=('se'))
    
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',col='cond',\
            row='order',data=norm_synch,kind='bar',errorbar=('se'))
    
    
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',
            order=[str(1),str(2),str(3)],errorbar=('se'),\
                kind='bar',col='cond',data=valid_synch.loc[valid_synch['order'].isin([3,4])])
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',
            order=[str(1),str(2),str(3)],errorbar=('se'),\
                kind='bar',col='cond',data=norm_synch.loc[norm_synch['order'].isin([3,4])])
    
    


norm_df['order'] = pd.to_numeric(norm_df['order'])
sns.catplot(x='applied_cluster',y='norm',hue='order',col='cond',\
            data=norm_df,kind='bar',errorbar=('se'))
    
norm_df_2 = pd.concat(norm_df_2)
norm_df_2['order'] = pd.to_numeric(norm_df_2['order'])
sns.catplot(x='applied_cluster',y='norm',hue='order',col='cond',\
            data=norm_df_2,kind='bar',errorbar=('se'))

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x='applied_cluster',y='norm',hue='cond',order=[str(1),str(2),str(3)],
            data=norm_df.loc[norm_df['order'].isin([1,2])],errorbar=('ci',68),ax=ax)
ax.axhline(1,ls=':',color='black',lw=2)
    
fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x='applied_cluster',y='norm',hue='cond',order=[str(1),str(2),str(3)],
            data=norm_df.loc[norm_df['order'].isin([3,4])],errorbar=('ci',68),ax=ax)
ax.axhline(1,ls=':',color='black',lw=2)




# Create dataframe to parallelize synchrony calculations 
pair_df = []
for idx,var in tqdm(out_df.groupby(['cond','animal']),position=0, leave=True, colour='green',\
                       desc = 'Creating dataframe for synchrony calculations'):
    
    # Initiat the figure
    fig,axes = plt.subplots(figsize=(35,20), ncols = len(merged_data.cluster_num.unique()),\
                   nrows = len(out_df.order.unique()), sharex = True, sharey = True)
    for order in sorted(var['order'].unique()):
        pos = order
        if order ==7:
            pos = 3
        if order ==8:
            pos =4
        
        dset = var.loc[(var['order']==order)]           
        applied = valid_df.loc[(valid_df['cond']==var['cond'].iloc[0])&\
                               (valid_df['animal']==var['animal'].iloc[0])&(valid_df['order']==order)]
        
        # Insert a column that indicates the cluster number that was created based
        # on whether the cell was stable or not
        # Note: 'cluster_pair' column will be the correct number (now)
        # to the 'applied_cluster' column
        dset.insert(5,'applied_cluster',value = '')
        for cell in dset.cell.unique():
            dset.loc[(dset.cell==cell), 'applied_cluster'] =\
                int(applied.loc[(applied.cell==cell)]['applied_cluster'].iloc[0]+1)
        dset['applied_cluster'] = dset['applied_cluster'].astype(int) 

        # Create the dataframe and raster plots
        for cluster in sorted(dset.applied_cluster.unique()):
            
            ################################
            # Extract the intra-cluster data
            ################################
            
            # Query out the data
            cset = dset.loc[(dset['applied_cluster']==cluster)]
            
            # Determine if there are more than one cell, if so
            # proceed with including in dataframe
            if len(cset['cell'].unique())>1:
                cell_pairs = list(combinations(cset['cell'].unique(), 2))
                
                for cell1,cell2 in cell_pairs:
                    df_out = pd.DataFrame(np.expand_dims(np.asarray([idx[0],idx[1],order,cluster,cluster,'intra',cell1,cell2]),0),\
                                          columns=['cond','animal','order','applied_cluster','compared_cluster','type','cell1','cell2'])
                    df_out[['cell1_phase','cell2_phase']] = [[cset.loc[cset['cell']==cell1]['instant_phase'].iloc[0],\
                                                                    cset.loc[cset['cell']==cell2]['instant_phase'].iloc[0]]]
                    
                    # ensure lengths are the same (sometimes off by 1-2 samples)
                    min_len = min(min([[cset.loc[cset['cell']==cell1]['instant_phase'].iloc[0].shape[0],\
                                    cset.loc[cset['cell']==cell2]['instant_phase'].iloc[0].shape[0]]]))
                    df_out['phase_diff'] = [np.mod(abs(df_out['cell1_phase'].iloc[0][:min_len]-\
                                            df_out['cell2_phase'].iloc[0][:min_len]),2*np.pi)]
                    pair_df.append(df_out)
            
            ################################
            # Extract the inter-cluster data
            ################################
            cset_out = dset.loc[(dset['applied_cluster']!=cluster)]   
            if ((len(cset['cell'].unique())>1) & (len(cset_out['cell'].unique())>1)):

                cell_pairs = list(itertools.product(*[cset['cell'].unique(),cset_out['cell'].unique()]))
                for cell1,cell2 in cell_pairs:
                    cluster_num = cset_out.loc[cset_out['cell']==cell2]['applied_cluster'].iloc[0]
                    df_out = pd.DataFrame(np.expand_dims(np.asarray([idx[0],idx[1],order,cluster,cluster_num,'inter',cell1,cell2]),0),\
                                          columns=['cond','animal','order','applied_cluster','compared_cluster','type','cell1','cell2'])
                    df_out[['cell1_phase','cell2_phase']] = [[cset.loc[cset['cell']==cell1]['instant_phase'].iloc[0],\
                                                                    cset_out.loc[cset_out['cell']==cell2]['instant_phase'].iloc[0]]]
                    # ensure lengths are the same (sometimes off by 1-2 samples)
                    min_len = min(min([[cset.loc[cset['cell']==cell1]['instant_phase'].iloc[0].shape[0],\
                                    cset_out.loc[cset_out['cell']==cell2]['instant_phase'].iloc[0].shape[0]]]))
                    df_out['phase_diff'] = [np.mod(abs(df_out['cell1_phase'].iloc[0][:min_len]-\
                                            df_out['cell2_phase'].iloc[0][:min_len]),2*np.pi)]
                    pair_df.append(df_out)
            
            # Create the rasters
            pivot = cset.pivot(index=["cell"], columns=["spike_time_sec"],values="event")
            times = np.asarray(pivot.columns)
            t_array = np.arange(0,times[-1]+10,1/srate)
            
            raster_array = []
            for cell in range(pivot.shape[0]):
                spikes = times[~np.isnan(np.asarray(pivot)[cell,:])]
                spike_idx = [np.where(np.round(t_array,2)==spikes[x])[0][0]\
                             for x in range(len(spikes))]
                spike_array = np.ones_like(t_array)
                spike_array[:] = np.nan
                spike_array[spike_idx] = cell
                raster_array.append(spike_array)
            raster_array = np.asarray(raster_array)
            
            # Plot the raster
            axes[pos-1,cluster-1].set_title('Cluster: %i, Recording: %i' %(cluster, pos),fontsize=12)
            for cell in range(raster_array.shape[0]):
                axes[pos-1,cluster-1].scatter(t_array,raster_array[cell,:],marker='|',s=50)
        
    # Adjust layout for aesthetics
    plt.tight_layout()
    plt.suptitle('%s: %s Raster Plots' %(idx[0],idx[1]),fontsize=20,fontweight='bold',y=1.05)
    plt.savefig('raster_%s_%s.pdf' %(idx[0],idx[1]), bbox_inches="tight")
    plt.close(fig)

# Concatenate dataframes
pair_df = pd.concat(pair_df)

# Parallelize the phase-synchrony function
phase_synch_df = applyParallel_synch(pair_df.groupby(['cond','animal', 'order', 'applied_cluster','type']), phase_synch_par)

# SAve the output
phase_synch_df.loc[phase_synch_df['order']==str(7),'order']=str(3)
phase_synch_df.loc[phase_synch_df['order']==str(8),'order']=str(4)
phase_agnostic_synch_df.loc[phase_agnostic_synch_df['order']==str(7),'order']=str(3)
phase_agnostic_synch_df.loc[phase_agnostic_synch_df['order']==str(8),'order']=str(4)
phase_synch_df.to_pickle('phase_synch_postGMM_%s.pickle' %(time.strftime("%Y%m%d")))


g = sns.catplot(x="applied_cluster", y="phase_synch", hue="order",ci=68,
    kind="bar",col='type',data=phase_synch_df, row='cond',
    hue_order =sorted(phase_synch_df['order'].unique()),
    order = sorted(phase_synch_df.applied_cluster.unique()),
    palette = sns.color_palette("crest",len(phase_synch_df['order'].unique())),
        height=5, aspect=1.1)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Phase-synchrony impact by Chronic/Acute Stress\nCluster-specific',y=1.03)
g.fig.savefig("phase_synchrony_raw.pdf", bbox_inches="tight")




for idx,var in phase_agnostic_synch_df.groupby(['cond','animal','order']):
    
    # pivot the data
    pivotset = pd.pivot(var,index=['cell1','applied_cluster'],columns=['compared_cluster','cell2'],values='phase_synch')
    
    # sort the dataframe by group
    df_pivot = pivotset.sort_values(by='applied_cluster', ascending=True).reindex(sorted(pivotset.columns),axis=1)

    # Set the figure size
    plt.figure(figsize=(10, 8))
    
    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivotset, annot=False, cmap='coolwarm')
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pivot, annot=False, cmap='coolwarm')
        
    test = var[['cell1','cell2','applied_cluster','phase_synch']]
    test.set_index('applied_cluster')
    df_pivot = test.transpose()
    corr_matrix = df_pivot.corr()
    
    # Set the figure size
    plt.figure(figsize=(10, 8))
    
    # Generate the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

    
    
   


# Take the average across animal, order, cluster comparisons , and type
phase_synch_mean = phase_synch_df.groupby(['cond','animal','order','applied_cluster',\
                    'compared_cluster','type'])['phase_synch'].mean().reset_index()
    
    
sns.catplot(x="applied_cluster", y="phase_synch", hue="order",ci=68,
    kind="bar",col='type',data=phase_synch_mean, row='cond',
    hue_order =sorted(phase_synch_mean['order'].unique()),
    order = sorted(phase_synch_mean.applied_cluster.unique()),
    palette = sns.color_palette("crest",len(phase_synch_mean['order'].unique())),
       height=5, aspect=1.1)

sns.catplot(x="applied_cluster", y="phase_synch", hue="type",ci=68,
    kind="bar",data=phase_synch_mean.loc[phase_synch_mean['order'].isin(['3','4'])], row='cond',
    hue_order =['inter','intra'],
    order = sorted(phase_synch_mean.applied_cluster.unique()),
    palette = sns.color_palette("crest",len(phase_synch_mean['order'].unique())),
       height=5, aspect=1.1)
    
  
sns.catplot(x="cond", y="phase_synch", hue="type",ci=68,
    kind="point",col='order',data=phase_synch_mean,
    col_order =sorted(phase_synch_mean['order'].unique()),
    palette = sns.color_palette("crest",len(phase_synch_mean['order'].unique())),
       height=5, aspect=1.1)

sns.catplot(x="cond", y="phase_synch", hue="type",ci=68,
    kind="point",col='order',data=phase_synch_mean,row='applied_cluster',
    col_order =sorted(phase_synch_mean['order'].unique()),
    palette = sns.color_palette("crest",len(phase_synch_mean['order'].unique())),
       height=5, aspect=1.1)


sns.catplot(x="type", y="phase_synch", hue="cond",ci=68,
    kind="point",col='order',data=phase_synch_mean,
    col_order =sorted(phase_synch_mean['order'].unique()),
    palette = sns.color_palette("crest",len(phase_synch_mean['order'].unique())),
       height=5, aspect=1.1)

sns.catplot(x="type", y="phase_synch", hue="cond",ci=68,
    kind="point",col='order',data=phase_synch_mean,row='applied_cluster',
    col_order =sorted(phase_synch_mean['order'].unique()),
    palette = sns.color_palette("crest",len(phase_synch_mean['order'].unique())),
       height=5, aspect=1.1)

g = sns.pairplot(phase_synch_mean[['phase_synch','type','order','cond']], hue='type',\
             palette=gmm_colors[:best],plot_kws={'alpha':0.4}, height=5, aspect=1.5)

# Divide all recordings ('orders') by week 1 (pre: order 1)
phase_synch_norm = phase_synch_mean.groupby(['animal','applied_cluster',\
                        'compared_cluster','type']).apply(synch_norm).reset_index(drop=True)
phase_synch_norm['norm_week'] = pd.to_numeric(phase_synch_norm['norm_week'])
phase_synch_norm['norm_pre'] = pd.to_numeric(phase_synch_norm['norm_pre'])
phase_synch_norm['norm_post'] = pd.to_numeric(phase_synch_norm['norm_post'])




# =============================================================================
#  Cluster synchrony
# =============================================================================
import os
import easygui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize


# Import the file
dir_name = easygui.diropenbox(title='Directory selection',\
                    msg = 'Choose the directory the contains the pickle')
os.chdir(dir_name)

pickles = [x for x in os.listdir() if 'pickle' in x][0]
phase_agnostic_synch_df = pd.read_pickle(pickles)

# Set normalize parameta
norm = 'yes'

# For each recording session, look at heatmap of synchrony
synch_value_df =[]; synch_norm_df =[]
for idx,df in phase_agnostic_synch_df.groupby(['cond','animal']):
  
    # Get max synch values
    max_val = np.round(max(df['phase_synch']),2)
    
    # Calculate mean values
    df.loc[df['applied_cluster']!=df['compared_cluster'],'type'] = 'inter'
    
    mean_synch = df.groupby(['cond','animal','order',\
                  'applied_cluster','type'])['phase_synch'].mean().reset_index()
    synch_value_df.append(mean_synch)
    
    # Initiate figure and colorbar
    fig,axes = plt.subplots(figsize=(12,12),nrows=2,ncols=2)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    axes_list = axes.flatten()
    
    for order_idx,var in df.groupby('order'):
        if order_idx == str(7):
            order_idx = 3
        if order_idx == str(8):
            order_idx = 4
        
        ax = axes_list[int(order_idx)-1]
        
        if norm == 'yes':
            var['phase_synch']=(var['phase_synch']-\
                    var['phase_synch'].min())/(var['phase_synch'].max()-var['phase_synch'].min())
            mean_synch = var.groupby(['cond','animal','order',\
                          'applied_cluster','type'])['phase_synch'].mean().reset_index()
            synch_norm_df.append(mean_synch)
                
            max_val = 1
        
        # pivot the data
        pivotset = pd.pivot(var,index=['cell1','applied_cluster'],\
                            columns=['compared_cluster','cell2'],values='phase_synch')
        
        # sort the data to show unity line
        pivotset = pivotset.sort_index(level=['applied_cluster','cell1'],ascending=[True,True])
        pivotset.sort_index(axis=1, level=[0, 1], ascending=[True, True], inplace=True)
            
        # count number of cells in each cluster
        clust_cnt = var.groupby(['applied_cluster'])['cell1'].nunique().reset_index()
        
        x_starts = list(np.cumsum(clust_cnt['cell1'])-clust_cnt['cell1'])
        x_ends = list(np.cumsum(clust_cnt['cell1']))
        x_starts.insert(len(x_starts),0); x_ends.insert(0,0)
        
    
        # Generate heatmap w/ boxes around intra-cluster comparisons
        sns.heatmap(pivotset, annot=False, cmap='coolwarm',ax=ax,\
                        vmin=0,vmax=max_val,\
                        cbar_ax=cbar_ax)
 
        for i in range(len(clust_cnt)):
            ax.plot([x_starts[i]+0.1,x_starts[i]+0.1],[x_ends[i],x_ends[i+1]],lw=3,color='black') #Left
            if i == 0:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i]+(0.1*i),x_ends[i]+(0.1*i)],lw=3,color='black') #Top
            else:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i],x_ends[i]],lw=3,color='black') #Top
            
            if i == len(clust_cnt)-1:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i+1]-(0.1*i),x_ends[i+1]-(0.1*i)],lw=3,color='black') #Bottom
            else:
                ax.plot([x_starts[i]+0.1,x_ends[i+1]],[x_ends[i+1],x_ends[i+1]],lw=3,color='black') #Bottom
            
            if i < len(clust_cnt)-1:
                ax.plot([x_ends[i+1]+0.1,x_ends[i+1]+0.1],[x_ends[i],x_ends[i+1]],lw=3,color='black') #Right
            else:
                ax.plot([x_ends[i+1]-(0.1*i),x_ends[i+1]-(0.1*i)],[x_ends[i],x_ends[i+1]],lw=3,color='black') #Right

    # Save the figure
    plt.suptitle('%s: %s - Phase synchrony' %(df['cond'].iloc[0],df['animal'].iloc[0]))
    fig.savefig('%s_%s_phase_synch.pdf' %(df['cond'].iloc[0],df['animal'].iloc[0]),bbox_inches = 'tight')
    plt.close(fig)

# Concatenate mean dataframes
synch_value_df = pd.concat(synch_value_df)
synch_norm_df = pd.concat(synch_norm_df)

# Query out datasets that have intra and inter for a given session
cnt_set = synch_value_df.groupby(['cond','animal','order',\
                    'applied_cluster'])['type'].count().reset_index(name='cnt')
valid_synch = []; norm_synch =[]
for idx,row in cnt_set.iterrows():
    if row['cnt']==2:
        valid_synch.append(synch_value_df.loc[(synch_value_df['cond']==row['cond'])&\
                  (synch_value_df['animal']==row['animal'])&\
                (synch_value_df['order']==row['order'])&\
                    (synch_value_df['applied_cluster']==row['applied_cluster'])])
        norm_synch.append(synch_norm_df.loc[(synch_norm_df['cond']==row['cond'])&\
                  (synch_norm_df['animal']==row['animal'])&\
                (synch_norm_df['order']==row['order'])&\
                    (synch_norm_df['applied_cluster']==row['applied_cluster'])])
valid_synch = pd.concat(valid_synch)
norm_synch = pd.concat(norm_synch)

# Update orders
valid_synch.loc[valid_synch['order']==str(7),'order']=3
valid_synch.loc[valid_synch['order']==str(8),'order']=4
norm_synch.loc[norm_synch['order']==str(7),'order']=3
norm_synch.loc[norm_synch['order']==str(8),'order']=4

# Plot results
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',col='cond',\
            row='order',data=valid_synch,kind='bar',errorbar=('se'))
    
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',col='cond',\
            row='order',data=norm_synch,kind='bar',errorbar=('se'))
    
    
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',
            order=[str(1),str(2),str(3)],errorbar=('se'),\
                kind='bar',col='cond',data=valid_synch.loc[valid_synch['order'].isin([3,4])])
sns.catplot(x='applied_cluster',y='phase_synch',hue='type',
            order=[str(1),str(2),str(3)],errorbar=('se'),\
                kind='bar',col='cond',data=norm_synch.loc[norm_synch['order'].isin([3,4])])
    
    

# divide each intra by inter to compare across sessions*animals
norm_df=[]; 
for idx,df in valid_synch.groupby(['cond','animal','order','applied_cluster']):
    tmp = df.loc[df['type']=='intra'].reset_index()
    tmp['norm'] = tmp['phase_synch']/np.mean(df.loc[df['type']=='inter']['phase_synch'])
    norm_df.append(tmp)

norm_df_2= []
for idx,df in norm_synch.groupby(['cond','animal','order','applied_cluster']):
    tmp = df.loc[df['type']=='intra'].reset_index()
    tmp['norm'] = tmp['phase_synch']/np.mean(df.loc[df['type']=='inter']['phase_synch'])
    norm_df_2.append(tmp)

norm_df = pd.concat(norm_df)
norm_df['order'] = pd.to_numeric(norm_df['order'])
sns.catplot(x='applied_cluster',y='norm',hue='order',col='cond',\
            data=norm_df,kind='bar',errorbar=('se'))
    
norm_df_2 = pd.concat(norm_df_2)
norm_df_2['order'] = pd.to_numeric(norm_df_2['order'])
sns.catplot(x='applied_cluster',y='norm',hue='order',col='cond',\
            data=norm_df_2,kind='bar',errorbar=('se'))

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x='applied_cluster',y='norm',hue='cond',order=[str(1),str(2),str(3)],
            data=norm_df.loc[norm_df['order'].isin([1,2])],errorbar=('ci',68),ax=ax)
ax.axhline(1,ls=':',color='black',lw=2)
    
fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x='applied_cluster',y='norm',hue='cond',order=[str(1),str(2),str(3)],
            data=norm_df.loc[norm_df['order'].isin([3,4])],errorbar=('ci',68),ax=ax)
ax.axhline(1,ls=':',color='black',lw=2)

norm_df['week']=1; norm_df.loc[norm_df['order'].isin([3,4]),'week']=4

g = sns.catplot(x='applied_cluster',y='norm',hue='week',kind='bar',\
        order=[str(1),str(2),str(3)],data=norm_df,errorbar=('ci',68),
        col='cond',palette=['slategray','crimson'],lw=3,\
            errcolor='black',edgecolor='black')
for ax in g.axes.ravel():
    ax.axhline(1,ls=':',color='black',lw=2)
    
fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x='applied_cluster',y='norm',hue='cond',order=[str(1),str(2),str(3)],
            data=norm_df.loc[norm_df['order'].isin([3,4])],errorbar=('ci',68),ax=ax)
ax.axhline(1,ls=':',color='black',lw=2)


