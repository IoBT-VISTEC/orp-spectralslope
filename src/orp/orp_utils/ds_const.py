class MESA:
    
    def __init__(self):
        self.DATASET_CSV = 'mesa-sleep-harmonized-dataset-0.5.0.csv'
        self.ORG_SUBJ_ID_COL = 'mesaid'
        
    def format_subj_id(self, dataset_df):
        return ['mesa-sleep-{:04d}'.format(s) for s in dataset_df[self.ORG_SUBJ_ID_COL].values]


class SHHS1:
    
    def __init__(self):
        self.DATASET_CSV = 'shhs1-harmonized-dataset-0.18.0.csv'
        self.ORG_SUBJ_ID_COL = 'nsrrid'
        
    def format_subj_id(self, dataset_df):
        return ['shhs1-{}'.format(s) for s in dataset_df[self.ORG_SUBJ_ID_COL].values]


class CHATBaseline:
    
    def __init__(self):
        self.DATASET_CSV = 'chat-baseline-harmonized-dataset-0.12.0.csv'
        self.ORG_SUBJ_ID_COL = 'nsrrid'
        
    def format_subj_id(self, dataset_df):
        return ['chat-baseline-{}'.format(s) for s in dataset_df[self.ORG_SUBJ_ID_COL].values]


class WSC:
    
    def __init__(self):
        # self.DATASET_CSV = 'wsc-dataset-0.6.0.csv'
        self.DATASET_CSV = 'wsc-harmonized-dataset-0.6.0.csv'
        self.ORG_SUBJ_ID_COL = 'wsc_id'
        self.VISIT_COL = 'wsc_vst'
        
    def format_subj_id(self, dataset_df):        
        return ['wsc-visit{}-{}-nsrr'.format(v, s) 
                for v, s in zip(
                    dataset_df[self.VISIT_COL].values,
                    dataset_df[self.ORG_SUBJ_ID_COL].values,
                )]


class Dataset:
    def get_ds_config(self, ds_name):
        if ds_name.lower() == 'mesa':
            return MESA()
        elif ds_name.lower() == 'shhs/shhs1':
            return SHHS1()
        elif ds_name.lower() == 'chat/baseline':
            return CHATBaseline()
        elif ds_name.lower() == 'wsc':
            return WSC()
        else:
            raise Exception('Invalid ds_name')
