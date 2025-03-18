#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from utils import (
    create_if_no_exist, 
    load_configs, 
    plot_hypnogram,
    intersect,
    load_json,
)

from orp_utils.power_utils import (
    get_power_spectrum
)
from orp_utils.const import *
from nsrr_utils import clip_by_lights_on_off

import numpy as np
from matplotlib import pyplot as plt
from pandas.testing import assert_frame_equal
from scipy import signal
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

orp_config_path = './'
orp_config_filename = 'orp_config.json'
orp_config_fullpath = pjoin(orp_config_path, orp_config_filename)
        
conf_obj = load_configs(orp_config_fullpath)
conf = conf_obj['general_config']
freq_conf = conf_obj['freq_config']
power_type = freq_conf.power_type

config_filename = 'config.json'
config_inpath = pjoin(ds_name, config_filename)
conf_obj = load_configs(config_inpath)
samp_conf = conf_obj['sample_config']
output_conf = conf_obj['output_config']

if power_type not in ALLOWED_POWER_TYPES:
    raise Exception('invalid `power_type`')

signal_epoch_length = conf.signal_epoch_length
orp_epoch_length = conf.orp_epoch_length
label_epoch_length = 1 
assert label_epoch_length == 1 # support only 1s

outdir = 'extracted'
version = f'{output_conf.version}'
outpath = pjoin(ds_name, outdir, version)

config_outpath = pjoin(outpath, config_filename)
orp_config_outpath = pjoin(outpath, orp_config_filename)

if os.path.exists(outpath):
    ans = input(f'Outpath {outpath} already exists. Continue (y/n)? : ')
    if ans != 'y':
        raise Exception(f'Remove {outpath} before continue.')
else:   
    os.makedirs(outpath)

def check_config(a_path, b_path):
    a = load_json(a_path)
    b = load_json(b_path)
    
    if not sorted(a.items()) == sorted(b.items()):
        print(a)
        print(b)
        raise Exception('Different configs are not allowed in the same version. Please remove the current version or change to version to run.')

if os.path.exists(config_outpath):
    check_config(config_inpath, config_outpath)
    print('Done verified config.')

if os.path.exists(orp_config_outpath):
    check_config(orp_config_fullpath, orp_config_outpath)
    print('Done verified orp_config.')

shutil.copyfile(config_inpath, config_outpath)
print('Copy config:', config_filename, '->', config_outpath)

shutil.copyfile(orp_config_fullpath, orp_config_outpath)
print('Copy config:', orp_config_fullpath, '->', orp_config_outpath)

fs = conf.fs
class_map = conf.class_map
preproc_path = pjoin(samp_conf.preproc_path, ds_name)

log_fname_wildcard = f'log-v*.log'
nfile = len(glob(pjoin(outpath, 'logs', log_fname_wildcard)))
print('nfile:', nfile)

log_fname = log_fname_wildcard.replace('*', f'{nfile+1}')
print('log_fname:', log_fname)

logger = Logger(outpath, 
                active_file=log_fname)

sublog_path = create_if_no_exist(outpath, 'logs', log_fname[:-len('.log')])


data_info_path = DATA_INFO_CSV.replace('<ds_name>', ds_name)
data_info = pd.read_csv(data_info_path)
data_info.drop(['Unnamed: 0'], axis=1, inplace=True)


SUBJ_ID = 'subj_id'
subj_path = samp_conf.subj_path.replace('<ds_name>', ds_name)
subjects_df = pd.read_csv(subj_path)
subjects_df.drop(['Unnamed: 0'], axis=1, inplace=True)

subjects = sorted(subjects_df[SUBJ_ID].values, reverse=False)
logger('subjects:', len(subjects), subjects)

channel = samp_conf.channel
x_path = pjoin(preproc_path, 'X', channel)
labels_path = pjoin(preproc_path, samp_conf.labels_dir)
stages_path = pjoin(labels_path, 'stages')
arousals_path = pjoin(labels_path, 'arousals')

try:
    logger('do_clip_by_lights_on_off:', samp_conf.do_clip_by_lights_on_off)
except Exception as e:
    raise Exception(f'{e} | Add `do_clip_by_lights_on_off` to your dataset config file.')

def subepochs(data, y_true_org, orp_epoch_sec, fs):
    
    nepochs_orginal = len(data)
    orp_epoch_length = (orp_epoch_sec * fs)
    nsubepochs = len(data[0]) // orp_epoch_length
    
    # print('orp_epoch_length:', orp_epoch_length)
    if len(data[0]) % orp_epoch_length != 0:
        raise Exception(f'Epoch length should be divided by orp_epoch_length: {len(data[0])} & {orp_epoch_length}')
        
    new_x, new_y = [], []
    for idx in range(nepochs_orginal):
        for sub_idx in range(nsubepochs):
            new_x.append(data[idx][sub_idx*orp_epoch_length: (sub_idx+1)*orp_epoch_length])
            new_y.append(y_true_org[idx])
            
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    
    return new_x, new_y

def group_stages_label(stages, orp_epoch_length):
    # stages = stage per each second
    # group every `orp_epoch_length` seconds => 1 label
    return np.array([stages[s_index] for s_index in range(0, len(stages), orp_epoch_length)])

def group_arousals_label(arousals, orp_epoch_length):
    # arousal if at least one sec is arousal
    res = []
    
    for s_index in range(0, len(arousals), orp_epoch_length):
        if any(arousals[s_index:s_index+orp_epoch_length] == AROUSAL):
            res.append(AROUSAL)
        else:
            res.append(NONAROUSAL)
            
    return np.array(res)

WAKE = [i for i in class_map if class_map[i]=="W"][0]

def relabel_to_awake_asleep(stages, arousals):
    wake_samples = stages == WAKE 
    arousal_samples = arousals == AROUSAL
    
    awake_asleep = []
    for s, a in zip(stages, arousals):
        if str(s) == WAKE or a == AROUSAL:
            st = "awake"
        else:
            st = "asleep"
        awake_asleep.append(st)
    
    return np.array(awake_asleep)

def append_labels_to_df(
    results, nsamples, subj_id, stages, arousals, ws_labels
    ):
    
    results[SUBJ_ID] = [subj] * nsamples
    results['sample_id'] = range(nsamples)
    results['stage'] = stages
    results['arousal'] = arousals
    results['wakesleep'] = ws_labels

    results_df = pd.DataFrame(results)
    print(results_df)
    assert len(results_df) == nsamples
    
    return results_df

ntimes = int(signal_epoch_length / label_epoch_length) # support only label_epoch_length = 1
print(f'signal_epoch_length is {ntimes} times higher than label_epoch_length')

csv_path = create_if_no_exist(outpath, channel, 'csv')
print('csv_path:', csv_path)

raw_path = create_if_no_exist(outpath, channel, 'raw')
print('raw_path:', raw_path)

for subj in subjects:
                                            
    subj_csv = pjoin(csv_path, f'{subj}.csv')    
    subj_raw = pjoin(raw_path, f'{subj}.npy')    
    if not os.path.exists(subj_csv) or not os.path.exists(subj_raw):

        ### Load original samples and stages ###
        data = np.load(pjoin(x_path, f'{subj}.npy'))
        y_original = np.load(pjoin(preproc_path, f'{subj}_TRUE.npy'))
        logger(subj, 'y_original:', len(y_original), np.unique(y_original, return_counts=True))
        assert len(data) == len(y_original)
        
        ### Load 1-s stages & arousal labels ###
        stages1s = np.load(pjoin(stages_path, f'{subj}.npy'))
        arousals1s = np.load(pjoin(arousals_path, f'{subj}.npy'))
        
        assert len(stages1s) == len(arousals1s)
        assert len(stages1s) == len(y_original) * ntimes
        
        ### Clip original samples (to get start and end index) ###
        if samp_conf.do_clip_by_lights_on_off:
            clip_results = clip_by_lights_on_off(ds_name, data, y_original,
                                                 data_info, subjects_df, 
                                                 subj, signal_epoch_length, logger)
            data, y_original = clip_results["data"], clip_results["y_true_org"]
            start_index, end_index = clip_results["start_index"], clip_results["end_index"]
        
            ### Clip labels as original samples ###
            start_index_times = int(start_index * ntimes)
            end_index_times = int(end_index * ntimes)
            stages1s = stages1s[start_index_times: end_index_times]
            arousals1s = arousals1s[start_index_times: end_index_times]
            assert len(stages1s) == len(arousals1s)
            assert len(stages1s) == len(y_original) * ntimes
        
        ### Validate stages between original and 1-s ###
        for s_index in range(0, len(y_original)):
            # print(s_index, y_original[s_index], stages1s[s_index*ntimes:(s_index*ntimes)+ntimes])
            assert (y_original[s_index] == stages1s[s_index*ntimes:(s_index*ntimes)+ntimes]).all()

        ### divide epoch to orp_epoch_length ###
        n_original_samples = len(data)
        data, y_original = subepochs(data, y_original, orp_epoch_length, fs)
        assert len(data) == len(y_original)
        assert len(data) == n_original_samples * (signal_epoch_length / orp_epoch_length)
        
        ### group labels to orp_epoch_length ###
        n_original_stages = len(stages1s)
        stages = group_stages_label(stages1s, orp_epoch_length)
        logger(f"stages: {np.unique(stages, return_counts=True)}")
        arousals = group_arousals_label(arousals1s, orp_epoch_length)
        logger(f"arousals: {np.unique(arousals, return_counts=True)}")
        del stages1s, arousals1s
        
        assert len(stages) == len(arousals) == len(data)
        assert len(stages) == n_original_stages / orp_epoch_length

        ### Validate stages between original and 1-s again after dividing ###
        assert all(y_original == stages)
        del y_original
        
        ### relabel to awake/asleep ###
        ws_labels = relabel_to_awake_asleep(stages, arousals)
        logger(f"awake-asleep: {np.unique(ws_labels, return_counts=True)}")
        assert len(ws_labels) == len(stages)
        
        ### save raw signals ###
        np.save(subj_raw, data)
        logger(f"saved raw signals of shape: {data.shape} to {subj_raw}")
        
        ### calculate FFT ###
        results = get_power_spectrum(data, orp_epoch_length, fs, freq_conf, logger)
        df = append_labels_to_df(
            results, len(data), subj, 
            stages, arousals, ws_labels
        )
        
        if os.path.exists(subj_csv):
            # verify if the data are still the same
            current_df = pd.read_csv(subj_csv)
            assert_frame_equal(current_df.drop('Unnamed: 0', axis=1), df)
            
            logger(f'done verified {subj_csv}.')
            del current_df
            
        else:    
            ### save csv ###
            df.to_csv(subj_csv)
            logger(f'saved to {subj_csv}.')
        
        del df, data, results, stages, arousals, ws_labels

    else:
        logger(subj_csv, '-- already done: SKIP')
