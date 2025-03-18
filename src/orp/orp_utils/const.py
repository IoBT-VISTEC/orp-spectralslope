from .general_tools import (
    combination
)

DATA_INFO_CSV = '../../data_info/<ds_name>/label-info.csv'

SUBJ_ID_COL = 'subj_id'
BIN_NUMBER_COL = 'bin_number'
VALUE_COL = 'value'
WAKE_SLEEP_COL = 'wakesleep'
AROUSAL_COL = 'arousal'
STAGE_COL = 'stage'
STAGE_STR_COL = 'stage_str'
DECILE_COL = 'decile'
SUBSEQ_AW_COL = 'subseq_aw'

AWAKE = 'awake'
ASLEEP = 'asleep'
AROUSAL = 'arousal'
NONAROUSAL = 'nonarousal'

TRAIN = 'train'
VAL = 'val'

ALL_SUBJ = 'all_subjects'
BY_SUBJ = 'by_subject'
VISUALIZE_RUNNING_TYPES = [ALL_SUBJ, BY_SUBJ]

# +
RELATIVE = 'relative'
ABSOLUTE = 'absolute'

ALLOWED_POWER_TYPES = [RELATIVE, ABSOLUTE]
# -


stages_order = ["W", "N1", "N2", "N3", "R"]
arousals_order = [AROUSAL, NONAROUSAL]
wakesleeps_order = [AWAKE, ASLEEP]

labels_order = {
    STAGE_COL: stages_order,
    AROUSAL_COL: arousals_order,
    WAKE_SLEEP_COL: wakesleeps_order,
}
labels_combination = {
    COL: combination(labels_order[COL], 2)
    for COL in labels_order
}
fig_width = {
    STAGE_COL: 10,
    AROUSAL_COL: 8,
    WAKE_SLEEP_COL: 8,
}

vars_col = [
    '0.01-49.9_Slope',
    '0.3-30_Slope',
    '0.3-35_Slope',
    '0.3-1_Slope',
    '1-4_Slope',
    '4-8_Slope',
    '8-13_Slope',
    '13-30_Slope',
    '13-35_Slope',
    '30-49.9_Slope',
    '30-45_Slope',
    '30-35_Slope',
    '0.33333333-2.5_Slope',
    '2.5-6.5_Slope',
    '7.33333333-14.0_Slope',
    '14.33333333-35.0_Slope',
    
    # required outlier handling before using PSD
#     '0.01-49.9_psd_osc',
#     '0.1-35_psd_osc',
#     '0.3-30_psd_osc',
#     '0.3-35_psd_osc',
#     '0.01-1_psd_osc',
#     '0.1-1_psd_osc',
#     '0.3-1_psd_osc',
#     '1-4_psd_osc',
#     '4-8_psd_osc',
#     '8-13_psd_osc',
#     '13-30_psd_osc',
#     '13-35_psd_osc',
#     '30-49.9_psd_osc',
#     '30-45_psd_osc',
#     '30-35_psd_osc',
]

# +
R2 = 'R2'
MAE = 'MAE'
MSE = 'MSE'
CROSS_CORR = 'cross_correlation'

ORP = 'orp'
ORP_inv = 'orp_inv'
