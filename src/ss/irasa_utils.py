# +
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
import math
import pandas as pd
from glob import glob
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
import shutil
from scipy.signal import welch

try:
    from .const import *
except:
    from const import *
    

import sys
sys.path.append('../')

from utils import (
    plot_hypnogram
)

# -



def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices = index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x = nan_helper(y)
        >>> y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        
    Reference:
        - https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def get_scaled_val(val, scaled_range: tuple = (0, 4)):
    scaler = MinMaxScaler(scaled_range)
    scaled_val = scaler.fit_transform(val.reshape(-1, 1))
    scaled_val = np.squeeze(scaled_val)
    
    return scaled_val


def clean_outliers(val):
    q1 = np.percentile(val, 25)
    q3 = np.percentile(val, 75)
    iqr_val = q3 - q1
    # print('iqr_val', iqr_val, q3, q1)

    clean_val = np.array([np.nan 
                          if v < q1 - (1.5 * iqr_val) or v > q3 + (1.5 * iqr_val)
                          else v 
                          for v in val])
    
    nans, x = nan_helper(clean_val)
    nan_index = np.where(nans)[0]
    # print('nan_index', nan_index)

    if len(nan_index) > 0:
        clean_val[nans] = np.interp(x(nans), x(~nans), clean_val[~nans])

    return nan_index, clean_val


def plot_slope_var(subj_fit_params, subj_id, var, scale, channel, interpolate_outliers):
    yt = subj_fit_params[STAGE_COL].values
    scaled_range = (min(yt), max(yt))
    
    title = subj_id + ' ' + channel + ' ' + var
    val = subj_fit_params[var].values

    if scale:
        scaled_val = get_scaled_val(val, scaled_range)
    else:
        scaled_val = val

    title += ' (Min-Max scaled) ' 
    plot_hypnogram(y1=yt, y2=scaled_val,
                   label_y1=STAGE_COL, label_y2=var,
                   title=title)

    if interpolate_outliers:
        nan_index, clean_val = clean_outliers(val)
        
        if len(nan_index) > 0:
            if scale:
                scaled_clean_val = get_scaled_val(clean_val, scaled_range)
            else:
                scaled_clean_val = clean_val
                
            title += ' - interpolate outliers' 
            plot_hypnogram(y1=yt, y2=scaled_clean_val,
                           label_y1=STAGE_COL, label_y2=var,
                           title=title,
                           vlines=nan_index,
                          )


def pandas_boxplot(subj_fit_params, columns, subj_id, channel, number_of_samples_per_class):
    fig, ax = plt.subplots(2, 2, sharey=False, figsize=(12, 8))
    bp = subj_fit_params[columns + [STAGE_COL]].boxplot(by=STAGE_COL, ax=ax)
    fig.tight_layout()
    fig.suptitle(f'{subj_id} {channel} | No. of samples: {len(subj_fit_params)}\n' +
                 f'{number_of_samples_per_class}',
                 y=1.04)
    fig.show()


def seaborns_plot(subj_fit_params, columns, subj_id, channel, 
                  number_of_samples_per_class, plot_type = ['box', 'swarm']):
    ncols = 2
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharey=False, figsize=(12, 10))
    order = [s for s in STAGES_ORDER if s in subj_fit_params[STAGE_COL].values]
    
    for ax_index, col in enumerate(columns):
        axi = int(ax_index // ncols)
        axj = int(ax_index % ncols)
        ax = axes[axi][axj]

        if 'box' in plot_type:
            sn.boxplot(x=STAGE_COL, y=col, data=subj_fit_params, ax=ax, order=order)
            
        if 'swarm' in plot_type:
            sn.swarmplot(x=STAGE_COL, y=col, data=subj_fit_params, color="orange", 
                         alpha=0.3, ax=ax, size=6, order=order)
            
        if 'violin' in plot_type:
            sn.violinplot(x=STAGE_COL, y=col, data=subj_fit_params, inner=None, 
                          ax=ax, edgecolor="gray", order=order)

    fig.suptitle(f'{subj_id} {channel} | No. of samples: {len(subj_fit_params)}\n' +
                 f'{number_of_samples_per_class}', 
                 y=1.04)
    fig.tight_layout()
    fig.show()


def plot_psd_by_class(all_psd, freqs, yt, ylabel, title, classes, class_map):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    assert len(all_psd) == len(yt)
    for cl in classes:
        all_psd_cl = all_psd[yt == cl]

        mean = np.mean(all_psd_cl, axis=0)
        std = np.std(all_psd_cl, axis=0)

        ax.plot(freqs, mean, label=class_map[str(cl)])
        ax.fill_between(freqs, mean-std, mean+std, alpha=0.5)
        ax.legend()
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Frequency (Hz)')

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.show()




