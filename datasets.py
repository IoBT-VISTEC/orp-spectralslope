import numpy as np
import os

subj_path = './subjects'
for ds_name in os.listdir(subj_path):
    if ds_name != '.DS_Store':
        for fname in os.listdir(os.path.join(subj_path, ds_name)):
            if ds_name != '.DS_Store':
                subjects = np.load(os.path.join(subj_path, ds_name, fname))
                print(subjects['train'])
                print(subjects['val'])