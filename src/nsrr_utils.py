from datetime import datetime, timedelta
import numpy as np

# +
DATE_FORMAT = '%y-%m-%d'
TIME_FORMAT = '%H:%M:%S'
date_today = datetime.strftime(datetime.now().date(), DATE_FORMAT)
date_tmr = datetime.strftime(datetime.now().date() + timedelta(days=1), DATE_FORMAT)

DATE_TIME_FORMAT = DATE_FORMAT + ' ' + TIME_FORMAT
# -

SUBJ_ID = 'subj_id'

def append_date_to_time(time):
    assert datetime.strptime(time, TIME_FORMAT)
    
    if int(time.split(':')[0]) >= 12: # %H
        date_time_str = date_today + ' ' + time
    else:
        date_time_str = date_tmr + ' ' + time
        
    return date_time_str


def clip_mesa(
        ds_name,
        data,
        y_true_org, 
        data_info,
        subjects_df,
        subj,
        nsec_per_epoch,
        logger
    ):
       
    n_original = len(data)
    
    ###########################
    ## Get time-related info ##
    ###########################
    
    # get recording start time
    recording_start = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'ststartp5'].values[0]

    # get lights on - off
    lights_off = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'stloutp5'].values[0]
    lights_on = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'stlonp5'].values[0]
    
    """
    # Not append fake date to datetime since it makes some subjects error
    # e.g., mesa-sleep-1866 who might have incorrect recording time (recorded 12:00 until 20:00)
    # and already checked that `lights_off_after_recording_seconds` computed correctly without date appended
    recording_start = append_date_to_time(recording_start)
    lights_off = append_date_to_time(lights_off)
    lights_on = append_date_to_time(lights_on)
    
    # convert time str to datetime
    recording_start_dt = datetime.strptime(recording_start, DATE_TIME_FORMAT)
    lights_off_dt = datetime.strptime(lights_off, DATE_TIME_FORMAT)
    """
    
    logger('recording_start:', recording_start, 
           '\nlights_off:', lights_off, '| lights_on:', lights_on)


    # convert time str to datetime
    recording_start_dt = datetime.strptime(recording_start, TIME_FORMAT)
    lights_off_dt = datetime.strptime(lights_off, TIME_FORMAT)
    
    
    # assert lights_off_dt >= recording_start_dt
    lights_off_after_recording_seconds = (lights_off_dt - recording_start_dt).seconds
    logger(f'Lights off after recording {lights_off_after_recording_seconds} seconds')

    # Time in Bed
    time_in_bed_mins = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'time_bed5'].values[0]
    time_in_bed_secs = time_in_bed_mins * 60
    logger('Time in Bed = {:.1f} seconds (~{:.4f} hours)'.format(
        time_in_bed_secs,
        time_in_bed_secs/60/60
    ))

    nepochs_in_bed = int(time_in_bed_secs / nsec_per_epoch)
    logger('nepochs in bed:', nepochs_in_bed)


    ###############################################################################
    ##                Cut X and y according to lights off - on                   ##
    ###############################################################################

    # At the begining
    if lights_off_after_recording_seconds > 0:
        start_epoch = int(lights_off_after_recording_seconds / nsec_per_epoch)
        logger(f'Clip {start_epoch} epochs out at the beginning..')
        data = data[start_epoch:]
        y_true_org = y_true_org[start_epoch:]
        assert len(data) == len(y_true_org)
    else:
        start_epoch = 0

    # At the end
    diff = len(data) - nepochs_in_bed
    if diff > 0:
        data = data[:-diff]
        y_true_org = y_true_org[:-diff]
        logger(f"Clip {diff} epochs out at the end..")

    elif len(data) == nepochs_in_bed:
        logger("No need to clip epochs at the end")

    else:
        logger("No need to clip epochs at the end.",
               "Some epochs are already excluded (Due to non-stages)")

        
    assert len(data) == len(y_true_org)
    logger(f'Finally, remains: {len(y_true_org)} epochs: {np.unique(y_true_org, return_counts=True)}')
    
    return {
        "data": data,
        "y_true_org": y_true_org,
        "start_index": start_epoch,
        "end_index": n_original - diff,
    }


def clip_shhs(
        ds_name,
        data,
        y_true_org, 
        data_info,
        subjects_df,
        subj,
        nsec_per_epoch,
        logger
    ):
    
    n_original = len(data)
    
    ###########################
    ## Clip at the beginning ##
    ###########################
    
    # get recording start time
    recording_start = data_info.loc[data_info[SUBJ_ID]==subj, 'record_start_datetime'].values[0]
    recording_start = recording_start.split(' ')[1].split('+')[0]
    
    # convert time str to datetime
    recording_start_dt = datetime.strptime(recording_start, TIME_FORMAT)
    
    
    ## 'stloutp' = the start of time in bed
    lout = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'stloutp'].values[0]
    if 'shhs1' in ds_name.lower():
        # SHHS1 lout = number of epochs after recording starts
        n_secs_before_lights_off = int(lout * nsec_per_epoch)
        lights_off = recording_start_dt + timedelta(seconds=n_secs_before_lights_off)
        logger('n_epochs_before_lights_off:', lout,
               '| n_secs_before_lights_off:', n_secs_before_lights_off,
               '| recording_start:', recording_start,
              )
    
    else: 
        # SHHS2 = clock time of lights off / start of time in bed
        lights_off = lout
        lights_off_dt = datetime.strptime(lights_off, TIME_FORMAT)
    
    # preprocessed data started from
    preproc_data_start_sec = int(data_info.loc[data_info[SUBJ_ID]==subj, 'label_start_second'].values[0])
    preproc_data_start = recording_start_dt + timedelta(seconds=preproc_data_start_sec)
    

    logger('recording_start:', recording_start, 
           'preproc_data_start_sec:', preproc_data_start_sec,
           'preproc_data_start:', preproc_data_start,
           '\nlights_off:', lights_off)


    if lights_off > preproc_data_start:
        # lights off after preproc_data_start
        time_diff_sec = (lights_off - preproc_data_start).seconds
        start_epoch = int(time_diff_sec / nsec_per_epoch)
        logger('clip at the beginning:', time_diff_sec, '| start from epoch:', start_epoch)
        
        """
        # I was going to verify whether the number of epochs before sleep onset are the same as in csv.
        # But, many subjects got 'sleep_latency' label as unreliable. So, It's not verified.
        WAKE = 0
        slp_onset_epoch = next((idx for idx, s in enumerate(y_true_org) if s != WAKE), -1)
        if slp_onset_epoch == -1:
            logger('This subject has been awake for the entire night.')
        else:
            sleep_onset_from_csv = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'slplatp'].values[0] # minute
            print('sleep_onset_from_csv:', sleep_onset_from_csv)
            print('slp_onset_epoch:', slp_onset_epoch)
            
            assert sleep_onset_from_csv == (slp_onset_epoch * nsec_per_epoch) / 60
        """
        
    else:
        # lights off before preproc_data_start => do nothing (start using when preproc_data starts)
        start_epoch = 0
    
    
    data = data[start_epoch:]
    y_true_org = y_true_org[start_epoch:]
    assert len(data) == len(y_true_org)
    
    
    #####################
    ## Clip at the end ##
    #####################
    

    # Time in Bed
    time_in_bed_mins = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'timebedp'].values[0]
    time_in_bed_secs = time_in_bed_mins * 60
    logger('Time in Bed = {:.1f} seconds (~{:.4f} hours)'.format(
        time_in_bed_secs,
        time_in_bed_secs/60/60
    ))

    nepochs_in_bed = int(time_in_bed_secs / nsec_per_epoch)
    logger('nepochs in bed:', nepochs_in_bed)


    # At the end
    diff = len(data) - nepochs_in_bed
    if diff > 0:
        data = data[:-diff]
        y_true_org = y_true_org[:-diff]
        logger(f"Clip {diff} epochs out at the end..")

    elif len(data) == nepochs_in_bed:
        logger("No need to clip epochs at the end")

    else:
        logger("No need to clip epochs at the end.",
               "Some epochs are already excluded (Due to non-stages)")

        
    assert len(data) == len(y_true_org)
    logger(f'Finally, remains: {len(y_true_org)} epochs: {np.unique(y_true_org, return_counts=True)}')
    
    return {
        "data": data,
        "y_true_org": y_true_org,
        "start_index": start_epoch,
        "end_index": n_original - diff,
    }


# +

def clip_chat(
        ds_name,
        data,
        y_true_org, 
        data_info,
        subjects_df,
        subj,
        nsec_per_epoch,
        logger
    ):
    
    n_original = len(data)
    
    ###########################
    ## Clip at the beginning ##
    ###########################
    
    # get recording start time
    recording_start = data_info.loc[data_info[SUBJ_ID]==subj, 'record_start_datetime'].values[0]
    recording_start = recording_start.split(' ')[1].split('+')[0]
    
    # convert time str to datetime
    recording_start_dt = datetime.strptime(recording_start, TIME_FORMAT)
    print(recording_start_dt)
    
    
    ## Calculate the start of time in bed
    # stlonp (e.g., 6:00 AM -- need to convert) - timebedp (mins)
    
    # Lights on
    lights_on_str = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'stlonp'].values[0]
    
    # convert time str to datetime
    lights_on_dt = datetime.strptime(lights_on_str, '%H:%M %p')
    print('lights_on_str:', lights_on_str, 'original:', lights_on_dt)
    
    if 'AM' in lights_on_str:
        lights_on_dt += timedelta(days = 1)
    print('lights_on_dt + 1 =', lights_on_dt)
    
    
    # Time in Bed
    time_in_bed_mins = subjects_df.loc[subjects_df[SUBJ_ID]==subj, 'timebedp'].values[0]
    
    # Lights off
    lights_off_dt = lights_on_dt - timedelta(minutes=time_in_bed_mins)
    logger('Lights off : {} Lights on : {} (Time in bed : {} hrs {} mins)'.format(
        lights_off_dt,
        lights_on_dt,
        time_in_bed_mins // 60,
        time_in_bed_mins % 60,
    ))
    
    time_in_bed_secs = time_in_bed_mins * 60
    logger('Time in Bed = {:.1f} seconds (~{:.4f} hours)'.format(
        time_in_bed_secs,
        time_in_bed_secs/60/60
    ))

    nepochs_in_bed = int(time_in_bed_secs / nsec_per_epoch)
    logger('nepochs in bed:', nepochs_in_bed)
    
    
    # preprocessed data started from
    preproc_data_start_sec = int(data_info.loc[data_info[SUBJ_ID]==subj, 'label_start_second'].values[0])
    preproc_data_start = recording_start_dt + timedelta(seconds=preproc_data_start_sec)
    

    logger('recording_start:', recording_start, 
           'preproc_data_start_sec:', preproc_data_start_sec,
           'preproc_data_start:', preproc_data_start)


    if lights_off_dt > preproc_data_start:
        # lights off after preproc_data_start
        time_diff_sec = (lights_off_dt - preproc_data_start).seconds
        start_epoch = int(time_diff_sec / nsec_per_epoch)
        logger('clip at the beginning:', time_diff_sec, '| start from epoch:', start_epoch)
        
        
    else:
        # lights off before preproc_data_start => do nothing (start using when preproc_data starts)
        start_epoch = 0
    
    
    data = data[start_epoch:]
    y_true_org = y_true_org[start_epoch:]
    assert len(data) == len(y_true_org)
    
    
    #####################
    ## Clip at the end ##
    ####################

    # At the end
    diff = len(data) - nepochs_in_bed
    if diff > 0:
        data = data[:-diff]
        y_true_org = y_true_org[:-diff]
        logger(f"Clip {diff} epochs out at the end..")

    elif len(data) == nepochs_in_bed:
        logger("No need to clip epochs at the end")

    else:
        logger("No need to clip epochs at the end.",
               "Some epochs are already excluded (Due to non-stages)")

        
    assert len(data) == len(y_true_org)
    logger(f'Finally, remains: {len(y_true_org)} epochs: {np.unique(y_true_org, return_counts=True)}')
    
    return {
        "data": data,
        "y_true_org": y_true_org,
        "start_index": start_epoch,
        "end_index": n_original - diff,
    }


# -

def clip_by_lights_on_off(
        ds_name,
        data,
        y_true_org, 
        data_info,
        subjects_df,
        subj,
        nsec_per_epoch,
        logger
    ):
    
    if 'mesa' in ds_name.lower():
        clip_fn = clip_mesa
    elif 'shhs' in ds_name.lower():
        clip_fn = clip_shhs
    elif 'chat' in ds_name.lower():
        clip_fn = clip_chat
    else:
        raise Exception(f'Define function to clip data from {ds_name} first.')
        
    return clip_fn(
        ds_name,
        data,
        y_true_org, 
        data_info,
        subjects_df,
        subj,
        nsec_per_epoch,
        logger,
    )

