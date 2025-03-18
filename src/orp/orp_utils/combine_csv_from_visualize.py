# # Notes:
# * Might consider not to use this code because the data will be duplicated and increase the chance to create bugs
# * The reproduce_orp, which requires the vars and labels from all subjects, is already able to read from each subject's folder (no need to run this code first)

# +
import os
from glob import glob
import pandas as pd
import numpy as np

pjoin = os.path.join
# -

csv_path = '../SHHS/shhs1/extracted/v3/EEG EEG/results/subjects-v1/visualize-orp-v1-irasa-v6-lookup-table-MESA-v1/val/all_subjects_30s'
SUBJ_ID_COL = 'subj_id'


subjects_dir = [subj for subj in os.listdir(csv_path) 
                if os.path.isdir(pjoin(csv_path, subj)) and not '.ipynb_checkpoints' in subj]
len(subjects_dir), subjects_dir

# get list of csv files
list_csv = [f for f in os.listdir(pjoin(csv_path, subjects_dir[0])) if '.csv' in f]
len(list_csv), list_csv

dict_nrows = {}
for f in list_csv:
    print('#'*30, f, '#'*30)
    
    all_subj_df = pd.DataFrame()
    cnt_subj = 0
    for subj in subjects_dir:
        subj_path = pjoin(csv_path, subj)
        print('='*50)
        print('subj_path:', subj_path)

        df = pd.read_csv(pjoin(subj_path, f))

        if subj in dict_nrows:
            if 'down' in f and '30s' in csv_path:
                assert dict_nrows[subj] == len(df) * 10
            else:
                assert dict_nrows[subj] == len(df)
        else:
            dict_nrows[subj] = len(df)

        all_subj_df = all_subj_df.append(df)
        cnt_subj += 1
        print('current len(df):', len(all_subj_df))
    
    assert all(np.unique(all_subj_df[SUBJ_ID_COL]) == np.unique(subjects_dir))
    
    outpath = pjoin(csv_path, f)
    all_subj_df.to_csv(outpath)
    print(f'Done saving {len(all_subj_df)} rows (from {cnt_subj} subjects) to: {outpath}')
    print('#'*50)
    
    del all_subj_df


test_df = pd.read_csv(outpath)
test_df


