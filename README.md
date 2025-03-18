# Comprehensive Evaluation of Odds Ratio Product and Spectral Slope as Continuous Sleep Depth Measures: Advancing Sleep Staging and Clinical Applications

This repository computes Odd Ratio Product (ORP) and Spectral Slope, which were comprehensively compared in the study.

## Datasets

Data in the study were available upon request in the National Sleep Research Resource (NSRR). Lists of subjects included in the study are in `src/orp/[DS_NAME]/subjects`.

## ORP (`src/orp`)

1. Configure ORP in `orp_config.json`
2. Configure each dataset's information in `[DS_NAME]/config.json`,
    - `subj_path` : a csv file from NSRR, containing a list of subjects to extract ORP. An example of CSV file (with only one example subject) is in `src/ss/MESA/subjects/EEG3-good-subjects.csv`.
    - `preproc_path` : path to preprocessed data
    - `labels_dir` : path to 1s sleep stage labels (extracted from 30s labels) and arousal labels 
    - `do_clip_by_lights_on_off` : whether or not the dataset should be cropped by lights off/on markers
3. Prepare `label-info.csv` (See examples in `/data_info/[DS_NAME]/label-info.csv`)
4. Run `python prepare_orp.py` to calculate power spectral density and label awake/asleep for each 3s epoch
5. Run `construct_lookup_table.ipynb` to construct a look-up table (when `dset` = `TRAIN`) and get ORP values (when `dset` = `VAL`)


## Spectral Slope (`src/ss`)

1. Configure Spectral Slope in `irasa_config.json`
2. Configure each dataset's information in `[DS_NAME]/config.json`,
3. Run `python IRASA.py` to compute spectral slope for each 3s epoch