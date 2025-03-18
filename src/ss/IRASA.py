#!/usr/bin/env python
# coding: utf-8

########################
###### References ######
########################

# IRASA - https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb

import sys
sys.path.append('../')

from utils import (
    create_if_no_exist, 
    load_configs, 
    plot_hypnogram
)

from irasa_utils import (
    plot_slope_var,
    pandas_boxplot,
    seaborns_plot,
    plot_psd_by_class,
)
from nsrr_utils import clip_by_lights_on_off
from const import *

import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams
import math
import pandas as pd
from glob import glob
import seaborn as sn
from tqdm import tqdm
from mpunet.logging import Logger
from sklearn.preprocessing import MinMaxScaler
import shutil
from scipy.signal import welch
import yasa
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(action='once')

import logging
logging.disable(logging.WARNING)

import os
pjoin = os.path.join

########################################
## load config and copy to output dir ##
########################################

ds_name = "MESA" # SHHS/shhs1, MESA, CHAT/baseline, WSC
save_spec = True

irasa_config_path = './'
irasa_config_filename = 'irasa_config.json'
irasa_config_fullpath = pjoin(irasa_config_path, irasa_config_filename)
        
conf_obj = load_configs(irasa_config_fullpath)
conf = conf_obj['general_config']
irasa_conf = conf_obj['irasa_config']

config_filename = 'config.json'
config_inpath = pjoin(ds_name, config_filename)
conf_obj = load_configs(config_inpath)
samp_conf = conf_obj['sample_config']
output_conf = conf_obj['output_config']

outdir = 'extracted'
version = f'{output_conf.version}'
outpath = pjoin(ds_name, outdir, version)

config_outpath = pjoin(outpath, config_filename)
irasa_config_outpath = pjoin(outpath, irasa_config_filename)

if os.path.exists(outpath):
    ans = input(f'Outpath {outpath} already exists. Continue (y/n)? : ')
    if ans != 'y':
        raise Exception(f'Remove {outpath} before continue.')
else:   
    os.makedirs(outpath)

shutil.copyfile(config_inpath, config_outpath)
print('Copy config:', config_filename, '->', config_outpath)

shutil.copyfile(irasa_config_fullpath, irasa_config_outpath)
print('Copy config:', irasa_config_fullpath, '->', irasa_config_outpath)

fs = conf.fs
class_map = conf.class_map
nsec_per_epoch = conf.nsec_per_epoch
irasa_epoch_length = irasa_conf.epoch_length
preproc_path = pjoin(samp_conf.preproc_path, ds_name)

log_fname_wildcard = f'log-v*.log'
nfile = len(glob(pjoin(outpath, 'logs', log_fname_wildcard)))
print('nfile:', nfile)

log_fname = log_fname_wildcard.replace('*', f'{nfile+1}')
print('log_fname:', log_fname)

logger = Logger(outpath, 
                active_file=log_fname)

sublog_path = create_if_no_exist(outpath, 'logs', log_fname[:-len('.log')])

data_info_path = samp_conf.data_info_path.replace('<ds_name>', ds_name)
data_info = pd.read_csv(data_info_path)
data_info.drop(['Unnamed: 0'], axis=1, inplace=True)
data_info

subj_path = samp_conf.subj_path.replace('<ds_name>', ds_name)
print(subj_path)

subjects_df = pd.read_csv(subj_path)
subjects_df.drop(['Unnamed: 0'], axis=1, inplace=True)

subjects = sorted(subjects_df[SUBJ_ID_COL].values, reverse=False)
logger('subjects:', len(subjects), subjects)

# subjects = subjects[0:30]
logger('run in this code => subjects:', len(subjects), subjects)

fit_params_csv_path = create_if_no_exist(outpath, 'fit_params')
x_path = pjoin(preproc_path, 'X')
preproc_channels = sorted(os.listdir(x_path))
logger(preproc_channels)

try:
    logger('do_clip_by_lights_on_off:', samp_conf.do_clip_by_lights_on_off)
except Exception as e:
    raise Exception(f'{e} | Add `do_clip_by_lights_on_off` to your dataset config file.')

spec_path = create_if_no_exist(outpath, 'data_spec')

def subepochs(data, y_true_org, irasa_epoch_length):
    
    nepochs_orginal = len(data)
    nsubepochs = len(data[0]) // irasa_epoch_length
    
    if len(data[0]) % irasa_epoch_length != 0:
        raise Exception(f'Epoch length should be divided by irasa_epoch_length: {len(data[0])} & {irasa_epoch_length}')
        
    new_x, new_y = [], []
    for idx in range(nepochs_orginal):
        for sub_idx in range(nsubepochs):
            new_x.append(data[idx][sub_idx*irasa_epoch_length: (sub_idx+1)*irasa_epoch_length])
            new_y.append(y_true_org[idx])
            
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    
    return new_x, new_y

def extract_irasa(subj_id, data, y_true_org, fs, ch, irasa_conf, plot=False):
    # plot = True -> plot each epoch's results
    
    bands = irasa_conf.frequency_range
    all_bands = {}
    for band in bands:
        all_bands['-'.join([str(b) for b in band])] = band
    
    kwargs_welch = irasa_conf.kwargs_welch
    window_size = irasa_conf.win_sec
    step_size_sec = irasa_conf.step_size_sec
    overlap = window_size - step_size_sec
    
    win = int(window_size * fs)
    nover = int(overlap * fs)
    
    kwargs_welch["noverlap"] = nover
    print(f'{subj_id} | compute_irasa..', f'win: {win}', f'bands: {bands}', kwargs_welch)
    all_sfreqs, all_psd_aperiodic, all_psd_osc, all_normal_psd = {}, {}, {}, {}
    
    fit_params_df = pd.DataFrame()
    for sample_idx, (d, y) in tqdm(enumerate(zip(data, y_true_org))):
        
        fit_params_sample_df = pd.DataFrame()
        for b_name in all_bands:
            band = all_bands[b_name]
            
            if not b_name in all_sfreqs:
                all_sfreqs[b_name] = []
                all_psd_aperiodic[b_name] = []
                all_psd_osc[b_name] = []
                all_normal_psd[b_name] = []
            
            sfreqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(d, fs,
                                                                    ch_names=ch, 
                                                                    band=band, 
                                                                    win_sec=window_size, 
                                                                    return_fit=True,
                                                                    kwargs_welch=kwargs_welch)

            # shape = (nchannels, nfreqs)
            psd_aperiodic = psd_aperiodic[0]
            psd_osc = psd_osc[0]
            assert len(sfreqs) == len(psd_aperiodic) == len(psd_osc)
            
            if sample_idx == 0:
                logger(b_name, ':', sfreqs[0], '-', sfreqs[-1])

            # Format fit_params
            fit_params = fit_params.drop('Chan', axis=1)
            
            # change all columns to attach with b_name
            cols_dict = {}
            for col in fit_params.columns:
                cols_dict[col] = b_name + '_' + col
            fit_params = fit_params.rename(columns=cols_dict)
            
            # sample_index and stage should add only once
            if not 'sample_index' in fit_params_sample_df.columns:
                fit_params['sample_index'] = sample_idx
                fit_params['stage'] = y
                    
            fit_params[b_name + '_sfreqs'] = str(sfreqs)
            
            # calculate PSD from periodic component
            power_osc = np.trapz(y=psd_osc, x=sfreqs)
            fit_params[b_name + '_psd_osc'] = power_osc
            
            # calculate PSD from normal PSD
            normal_psd = psd_aperiodic + psd_osc
            power = np.trapz(y=normal_psd, x=sfreqs)
            fit_params[b_name + '_psd'] = power
            
            
            fit_params_sample_df = pd.concat([fit_params_sample_df, fit_params], axis=1)
            all_sfreqs[b_name].append(sfreqs)
            all_psd_aperiodic[b_name].append(psd_aperiodic)
            all_psd_osc[b_name].append(psd_osc)
            all_normal_psd[b_name].append(normal_psd)
            
            if sample_idx % 1000 == 0:
                fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                ax.plot(d)
                plt.show()

                if len(sfreqs) > 20:
                    steps = 10
                else:
                    steps = 1
                    
                fig, ax = plt.subplots(1, 3, figsize=(15, 3))
                ax[0].plot(normal_psd)
                ax[0].set_xticks(range(0, len(sfreqs), steps))
                ax[0].set_xticklabels(sfreqs[range(0, len(sfreqs), steps)], rotation=90)
                ax[0].set_title('PSD')

                ax[1].plot(np.log(psd_aperiodic))
                ax[1].set_xticks(range(0, len(sfreqs), steps))
                ax[1].set_xticklabels(sfreqs[range(0, len(sfreqs), steps)], rotation=90)
                ax[1].set_title('(log) psd_aperiodic')

                ax[2].plot(psd_osc)
                ax[2].set_xticks(range(0, len(sfreqs), steps))
                ax[2].set_xticklabels(sfreqs[range(0, len(sfreqs), steps)], rotation=90)
                ax[2].set_title('psd_osc')

                fig.suptitle("{} (sample_index: {}) - {} | Slope (Aperi): {:.4f} PSD (OSC): {:.4f} PSD: {:.4f}".format(
                    subj_id,
                    sample_idx,
                    class_map[str(y)],
                    fit_params[b_name + '_Slope'].values[0],
                    fit_params[b_name + '_psd_osc'].values[0],
                    fit_params[b_name + '_psd'].values[0],
                ), y=1.02)
                plt.show()
                
            del sfreqs, psd_aperiodic, psd_osc, normal_psd
           
        fit_params_df = fit_params_df.append(fit_params_sample_df)
        del fit_params_sample_df
        
        
    # display(fit_params_df.head())
    for b_name in all_bands:
        print(b_name, len(fit_params_df), len(data), len(all_sfreqs[b_name]))
        assert len(fit_params_df) == len(data) == len(all_sfreqs[b_name])
        assert len(all_psd_aperiodic[b_name]) == len(all_psd_osc[b_name]) == len(all_sfreqs[b_name])
    
    
    return all_sfreqs, all_psd_aperiodic, all_psd_osc, all_normal_psd, fit_params_df


start = False
for subj in subjects:
    
    for ch in preproc_channels:
        # TODO requires to change manually for every dataset
        if 'SHHS/shhs1' in ds_name:
            if not 'EEG EEG' in ch: 
                continue
            elif 'sec' in ch:
                continue
        
        elif 'MESA' in ds_name:
            if not 'EEG EEG3' in ch: 
                continue
                
        elif 'CHAT' in ds_name:
            if not 'C4' in ch: 
                continue
                
        elif 'WSC' in ds_name:
            if not 'C3_M2' in ch: 
                continue
                
        else:
            raise Exception('Specify the rule (channels) to run first.')
        
        spec_ch_path = create_if_no_exist(spec_path, ch)
        
        try:
            csv_path = create_if_no_exist(fit_params_csv_path, ch)
            subj_csv = pjoin(csv_path, f'{subj}.csv')

            if not os.path.exists(subj_csv):
            
                y_true_org = np.load(pjoin(preproc_path, f'{subj}_TRUE.npy'))
                logger(subj, 'y_true_org:', len(y_true_org), np.unique(y_true_org, return_counts=True))

                data = np.load(pjoin(x_path, ch, f'{subj}.npy'))
                assert len(data) == len(y_true_org)
                
                """
                fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                ax.plot(y_true_org)
                ax.set_title(f'{subj} {ch}')
                plt.show()
                """

                if samp_conf.do_clip_by_lights_on_off:
                    # No clipping applied (for SHHS) due to unreliable labels
                    clip_results = clip_by_lights_on_off(ds_name, data, y_true_org,
                                                         data_info, subjects_df, 
                                                         subj, nsec_per_epoch, logger)
                    data, y_true_org = clip_results["data"], clip_results["y_true_org"]
                
                """
                fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                ax.plot(y_true_org)
                ax.set_title(f'{subj} {ch}')
                plt.show()

                plot_idx = 1
                fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                ax.plot(data[plot_idx])
                ax.set_title(f'{subj} {ch}')
                plt.show()
                """
            
                if irasa_epoch_length != fs * nsec_per_epoch:
                    data, y_true_org = subepochs(data, y_true_org, irasa_epoch_length)

                    """
                    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                    ax.plot(y_true_org)
                    ax.set_title(f'{subj} {ch} (after subepochs)')
                    plt.show()

                    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
                    ax.plot(data[plot_idx * len(data[0]//irasa_epoch_length)])
                    ax.set_title(f'{subj} {ch} (after subepochs)')
                    plt.show()
                    """

                assert len(data) == len(y_true_org)
                sfreqs, psd_aperiodic, psd_osc, psd, fit_params = extract_irasa(subj, data, y_true_org, 
                                                                                fs, ch, irasa_conf)
                fit_params.to_csv(subj_csv, index=['sample_index'])
                assert len(data) == len(y_true_org) == len(fit_params)
               
                if save_spec:
                    subj_spec_path = pjoin(spec_ch_path, f'{subj}.npz')
                    np.savez(subj_spec_path,
                             sfreqs = sfreqs, 
                             psd_aperiodic = psd_aperiodic,
                             psd_osc = psd_osc,
                             psd = psd,
                             stages = y_true_org,
                             x = data,
                            )
                    print(f'saved spec to {subj_spec_path}')
                    
                del y_true_org, data, sfreqs, psd_aperiodic, psd_osc, fit_params

            else:
                print(subj_csv, 'Already done: SKIP')
                
        except Exception as e:
            logger("ERROR", subj, ch)
            logger(e)
            logger('='*20, '\n\n')
