# +
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score
import math
import scipy
import shutil
from glob import glob
import os
pjoin = os.path.join

import json


# -

def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
        return obj


def save_json(json_dict, fpath, overwrite = None):
    
    if overwrite is not True:
        overwrite_str = 'o'
        if os.path.exists(fpath):
            overwrite_str = str(input(f'{fpath} already exists. overwrite (o) / append (a) / stop (s)? : '))

        if overwrite_str == 'a':
            # append to the current json file
            current_dict = load_json(fpath)
            json_dict.update(current_dict)
            print('append to current dict => dict merged:', json_dict)

        elif overwrite_str == 's':
            raise Exception('Stop due to the selected option. Change filename manually before re-running.')

        elif overwrite_str not in ['o', 'a', 's']:
            raise Exception('Invalid option')
        
    with open(fpath, "w") as outfile:
        json.dump(json_dict, outfile, indent=4)
        print('saved dict to:', fpath)


def intersect(lst):
    return sorted(set(lst[0]).intersection(*lst[1:]))


def create_if_no_exist(*args):
    pname = pjoin(*args)
    if not os.path.exists(pname):
        os.makedirs(pname)
        print(f'created: {pname}')
        
    return pname


def remove_and_create_if_exist(*args):
    outpath = pjoin(*args)
    if os.path.exists(outpath):
        ans = input(f'{outpath} already exists. Remove and Overwrite (y/n)? : ')
        if ans == 'y':
            shutil.rmtree(outpath)
            print(f'removed: {outpath}')

            print('created', outpath)
            os.makedirs(outpath)
            
        else:
            ans = input(f'Continue without removing (y/n)? : ')
            if ans != 'y':
                raise Exception(f'{outpath} is not removed. Please re-config before continue.')
            else:
                print(f'{outpath} is not removed and continue..')
    else:
        os.makedirs(outpath)
        
    return outpath


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_configs(config_filename):
    config_json = load_json(config_filename)
    
    conf = {}
    for key in config_json.keys():
        conf[key] = Config(**config_json[key])
    
    return conf


def get_nsamples_per_class(y):
    cl, ncl = np.unique(y, return_counts=True)
    return [f'{c}: {n}' for c, n in zip(cl, ncl)]


def plot_hypnogram(y1=[], y2=[], y3=[],
                   label_y1='', label_y2='', label_y3='',
                   title='', vlines=[], figpath='', class_map={}):
    
    assert len(y1) > 0 or len(y2) > 0 or len(y3) > 0
    
    fig, ax = plt.subplots(figsize=(15, 3))
    
    if len(y1) > 0:
        ax.plot(y1, label=label_y1, color='#9DC3DC')

    if len(y2) > 0:
        ax.plot(y2, label=label_y2, color='#5278B7')
        
    if len(y3) > 0:
        ax.plot(y3, label=label_y3, color='black', alpha = 0.4)
        
    
    if len(class_map) > 0:
        stages_label_idx = [int(k) for k in class_map.keys()]
        stages_label_name = list(class_map.values())
        
        ax.set_yticks(stages_label_idx)
        ax.set_yticklabels(stages_label_name)
        
        
    if len(vlines) > 0:
        for v in vlines:
            ax.axvline(v, alpha=0.3, color='grey')
    
    ax.set_title(title)
    # plt.legend()
    if len(figpath) == 0:
        plt.show()
    else:
        plt.savefig(figpath)


def get_class_map_from_list(class_list):
    class_map = {}
    for cl_idx, cl in enumerate(class_list):
        class_map[str(cl_idx)] = cl
    return class_map


def get_list_from_class_map(class_map):
    arr = [''] * len(list(class_map.keys()))
    for k in class_map:
        arr[int(k)] = class_map[k]
    return arr


