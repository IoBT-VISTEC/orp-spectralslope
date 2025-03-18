import numpy as np
import pandas as pd
import math
import json
from tqdm import tqdm
import seaborn as sns
from statannotations.Annotator import Annotator


from matplotlib import pyplot as plt
plt.ioff()

import os
pjoin = os.path.join

import sys
sys.path.append('../../')
sys.path.append('../../../')


from utils import (
    create_if_no_exist, 
    load_configs, 
    plot_hypnogram
)
from .const import *
from .general_tools import (
    combination,
    Struct,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from scipy.signal import (
    correlate,
)
from scipy.stats import linregress





def get_inpath(
        dset,
        ds_name,
        channel,
        irasa_path,
        orp_version,
        subject_version,
        subj_lookup_table,
        lookup_table_config,
    ):
    
    #### inpath #####
    orp_path = f'{ds_name}/extracted/{orp_version}/{channel}'
    irasa_fit_params_path = pjoin(irasa_path, f'fit_params/{channel}')
    irasa_config_path = pjoin(irasa_path, 'irasa_config.json')

    results_path = pjoin(orp_path, f'results/subjects-{subject_version}')
    lookup_table_version = lookup_table_config['lookup_table_version']
    if subj_lookup_table == ALL_SUBJ:
        rank_power_path = pjoin(results_path, f'lookup-table-{lookup_table_version}') 
    else:
        rank_power_path = pjoin(results_path, subj_lookup_table, f'lookup-table-{lookup_table_version}') 
 
    rank_csv = pjoin(rank_power_path, f'rank_powers_{dset}.csv')
        
    lookup_table_path = pjoin(
        lookup_table_config['ds_name'],
        'extracted',
        orp_version,
        lookup_table_config['channel'],
        'results',
        f"subjects-{lookup_table_config['subject_version']}",
        f"lookup-table-{lookup_table_version}"
    )
    lookup_table_json = pjoin(lookup_table_path, 'lookup-table.json')
    lookup_table_version = f"{lookup_table_config['ds_name']}" + \
                           f"-subj{lookup_table_config['subject_version']}" + \
                           f"-{lookup_table_version}"
        
    
    return Struct(**{
        'irasa_fit_params_path': irasa_fit_params_path,
        'irasa_config_path': irasa_config_path,
        'results_path': results_path,
        'rank_power_path': rank_power_path,
        'lookup_table_path': lookup_table_path,
        'rank_csv': rank_csv,
        'lookup_table_json': lookup_table_json,
        'lookup_table_version': lookup_table_version,
    })


def get_outpath(
        dset,
        results_path,
        lookup_table_version,
        visualize_version,
        irasa_version,
        subj_lookup_table,
        target_epoch_duration,
    ):
    
    #### outpath #####
    visualize_dir = f'visualize-orp-{visualize_version}-irasa-{irasa_version}-lookup-table-{lookup_table_version}'
    if subj_lookup_table == ALL_SUBJ:
        visualize_path = pjoin(results_path, visualize_dir, dset) 
    else:
        visualize_path = pjoin(results_path, subj_lookup_table, visualize_dir) 
    print(f'output path : {visualize_path}')

    corr_by_subj_csv = pjoin(visualize_path, f'corr_by_subj_{target_epoch_duration}s.csv')
    figpath = pjoin(visualize_path, 'figure', f'{target_epoch_duration}s') 
    all_subj_path = pjoin(visualize_path, f'all_subjects_{target_epoch_duration}s') 
    corr_all_subj_csv = pjoin(all_subj_path, f'corr.csv')
    
    for p in [visualize_path, all_subj_path, figpath]:
        create_if_no_exist(p)
        
    return Struct(**{
        'visualize_path': visualize_path,
        'corr_by_subj_csv': corr_by_subj_csv,
        'figpath': figpath,
        'all_subj_path': all_subj_path,
        'corr_all_subj_csv': corr_all_subj_csv,
    })


def get_config(irasa_config_path):
    
    conf_obj = load_configs(irasa_config_path)
    conf = conf_obj['general_config']
    irasa_conf = conf_obj['irasa_config']

    fs = conf.fs
    irasa_epoch_length = irasa_conf.epoch_length
    print('irasa_epoch_length:', irasa_epoch_length)
    
    return Struct(**{
        'fs': fs,
        'irasa_epoch_length': irasa_epoch_length,
        'CLASS_MAP': conf.class_map,
    })


def get_epoch_config(
        orp_epoch_length, 
        irasa_epoch_length, 
        target_epoch_duration,
        fs
    ):
    
    print('orp_epoch_length:', orp_epoch_length)

    target_epoch_length = target_epoch_duration * fs
    print('target_epoch_length:', target_epoch_length)

    assert target_epoch_length % orp_epoch_length == 0
    assert target_epoch_length % irasa_epoch_length == 0
    average_orp_every = target_epoch_length // orp_epoch_length
    average_irasa_every = target_epoch_length // irasa_epoch_length

    print('average_orp_every:', average_orp_every)
    print('average_irasa_every:', average_irasa_every)
    
    return (
        target_epoch_length,
        average_orp_every,
        average_irasa_every,   
    )


def boxplot(var_name, value, stages):
    df = pd.DataFrame({
        STAGE_COL: stages,
        var_name: value,
    })
    ax = sns.boxplot(x=STAGE_COL, y=var_name, data=df, order=stages_order)
    ax = sns.swarmplot(x=STAGE_COL, y=var_name, data=df, 
                       color="grey", alpha=.25, order=stages_order)
    ax.set_title(f'{subj_id}')
    plt.show()


def cross_correlation(x, y):
    # https://scicoding.com/cross-correlation-in-python-3-popular-packages/
    
    nx = x - np.mean(x) # Demean
    ny = y - np.mean(y) # Demean
    
    if np.std(x) == 0:
        raise Exception(f'stdev should not be 0 => x: {len(x)} | std: {np.std(x)}')
    if np.std(y) == 0:
        raise Exception(f'stdev should not be 0 => y: {len(y)} | std: {np.std(y)}')

    corr = np.correlate(a=nx, v=ny)
    corr /= (len(y) * np.std(x) * np.std(y)) # Normalization
    assert len(corr) == 1
    
    """
    # To make sure the normalization method
    corr_stats = sm.tsa.stattools.ccf(y, x, adjusted=False)
    corr_stats = corr_stats[0:(len(y)+1)][0]
    assert np.isclose(corr[0], corr_stats)
    """
    
    return corr[0]


def r2_score(x, y):
    result = linregress(x, y)
    return {
        f'{R2}': result.rvalue**2,
        f'{R2}_slope': result.slope,
        f'{R2}_intercept': result.intercept,
        f'{R2}_rvalue': result.rvalue, # equals to cross correlation
        f'{R2}_pvalue': result.pvalue,
        f'{R2}_stderr': result.stderr,
    }


def group_stages_label(stages, average_every):
    # stages = stage per each second
    # group every `average_every` seconds => 1 label
    
    res = np.array([stages[s_index] for s_index in range(0, len(stages), average_every)])    
    return res


def group_arousals_label(arousals, average_every):
    # arousal if at least one sec is arousal
    res = []
    
    for s_index in range(0, len(arousals), average_every):
        if any(arousals[s_index:s_index+average_every] == AROUSAL):
            res.append(AROUSAL)
        else:
            res.append(NONAROUSAL)
            
    return np.array(res)


def group_wakesleeps_label(wakesleeps, average_every):
    # awake if at least one sec is awake
    res = []
    
    for s_index in range(0, len(wakesleeps), average_every):
        if any(wakesleeps[s_index:s_index+average_every] == AWAKE):
            res.append(AWAKE)
        else:
            res.append(ASLEEP)
    
    return np.array(res)


def cal_dist(x, y, metric, var1_name, var2_name):
    if metric == R2:
        dist = r2_score(x, y)
    elif metric == MAE:
        dist = { metric: mean_absolute_error(x, y) }
    elif metric == MSE:
        dist = { metric: mean_squared_error(x, y) }
    elif metric == CROSS_CORR:
        set_y = set(y) 
        if len(set_y) > 1:
            corr = cross_correlation(x, y)
            dist = { metric: corr }
        else:
            # print(f'{metric} was not calculated since y consists of 1 class only ({set_y})')
            return {}
    else:
        raise Exception(f'invalid metric: {metric}')
        
    results_dist = {f'{var1_name}-{var2_name}_{k}':v for k, v in dist.items()}
    return results_dist


def cal_stats(var, var_name):
    result_dict = {
        "mean": np.mean(var),
        "variance": np.var(var),
        "median": np.median(var),
    }
    result_dict = {var_name+'_'+k: v for k, v in result_dict.items()}
    return result_dict


def calculate_corr(
        subj_id,
        var1,
        var1_name,
        var1_compute_stats,
        var2,
        var2_name,
        var2_compute_stats,
        labels_list,
        prefix,
        figpath,
        plot,
    ):
    
    # print('calculate_corr..', var_name, labels_list, prefix)
    
    assert len(var1) == len(var2)
    assert all([len(var1) == len(labels_list[k]) for k in labels_list])
    result_dict = {}
    
    metrics = [
        CROSS_CORR,
        MAE,
        MSE,
        R2,
    ]
    
    # compute distance between vars (e.g., orp v.s. var) without classes
    for metric in metrics:
        results_dist = cal_dist(var1, var2, metric, var1_name, var2_name)
        if len(results_dist) > 0:
            result_dict.update(**results_dist)
            del results_dist
      
    # calculate stats in each var
    if var1_compute_stats:
        stats = cal_stats(var1, var1_name)
        result_dict.update(**stats)
        
    if var2_compute_stats:
        stats = cal_stats(var2, var2_name)
        result_dict.update(**stats)
    
    
    # compute metric within each class
    for labels_name in labels_list:
        labels = labels_list[labels_name]
        classes, n_per_class = np.unique(labels, return_counts=True)
            
        for cl, n_cl in zip(classes, n_per_class):
            result_dict['#'+cl] = n_cl
            
            select = labels == cl
            var1_this_cl = var1[select]
            var2_this_cl = var2[select]
            assert len(var1_this_cl) == len(var2_this_cl) == n_cl
        
            # distance between vars
            for metric in metrics:
                results_dist = cal_dist(var1_this_cl, var2_this_cl, metric, var1_name, var2_name)
                if len(results_dist) > 0:
                    result_dict.update(**{f'{k}|{labels_name}={cl}': v for k, v in results_dist.items()})
                    del results_dist
        
            # stats in each var
            if var1_compute_stats:
                stats = cal_stats(var1_this_cl, f'{var1_name}|{labels_name}={cl}')
                result_dict.update(**stats)
            if var2_compute_stats:
                stats = cal_stats(var2_this_cl, f'{var2_name}|{labels_name}={cl}')
                result_dict.update(**stats)
            
        
        # only done when both vars are in the same range (otherwise, it will always be sig diff)
        if 'scaled' in prefix:
            
            if var1_name != STAGE_COL and var2_name != STAGE_COL:
                # compute stats difference between vars (orp v.s. var)
                stat_results = compare_between_vars_groupby_labels(
                    subj_id,
                    var1,
                    var1_name,
                    var2,
                    var2_name,
                    labels,
                    labels_name,
                    figpath,
                    plot,
                )
                result_dict.update(**stat_results)
            
            if var1_compute_stats:
                # compute stats difference within var1 (e.g., orp) between two classes
                stat_results = compare_one_var_groupby_labels(
                    subj_id,
                    var1,
                    var1_name,
                    labels,
                    labels_name,
                    figpath,
                    plot,
                )
                result_dict.update(**stat_results)
            
            if var2_compute_stats:
                # compute stats difference within vars between two classes
                stat_results = compare_one_var_groupby_labels(
                    subj_id,
                    var2,
                    var2_name,
                    labels,
                    labels_name,
                    figpath,
                    plot,
                )
                result_dict.update(**stat_results)
            
        del labels
    
    # append prefix to dict key name
    result_dict = {f'{prefix}|{k}': v for k, v in result_dict.items()}
    return result_dict    


def save_for_computing_all_subjects(all_subj_path, arr_to_save, fname, subj_id):
    fname += '.csv'
    fpath = pjoin(all_subj_path, subj_id, fname)
    
    df = pd.DataFrame({
        SUBJ_ID_COL: [subj_id] * len(arr_to_save),
        VALUE_COL: arr_to_save
    })

    df.to_csv(fpath)
    print(f'{subj_id} | saved {len(arr_to_save)} rows to {fname}')


def read_all_subjects_values(all_subj_path, fname, subjects, stages):
    
    print('reading..', all_subj_path)
    df = pd.DataFrame()
    
    for subj_id in tqdm(subjects):
        df_tmp = pd.read_csv(pjoin(all_subj_path, subj_id, fname+'.csv'))
        df = df.append(df_tmp)
        del df_tmp
    
    values = df[VALUE_COL].values
    
    saved_subjects = pd.unique(df[SUBJ_ID_COL].values)
    assert all([s in subjects for s in saved_subjects])
    assert len(stages) == len(values)
    
    return values   


# +
def plot_stats_test(
        hue_plot_params, 
        pairs, 
        plot, 
        subj_id, 
        title, 
        fwidth, 
        figpath, 
        var_name,
    ):
    
    results = {}
    # print('<plot_stats_test>', hue_plot_params, pairs, plot, subj_id, title, fwidth, var_name)
    
    with sns.plotting_context("notebook", font_scale = 1.1):
        # Create new plot
        fig, ax = plt.subplots(figsize=(fwidth, 6))

        # Plot with seaborn
        ax = sns.boxplot(ax=ax, **hue_plot_params)

        # Add annotations
        if len(pairs) > 0:
            annotator = Annotator(ax, pairs, **hue_plot_params, verbose=False)
            annotator.configure(test="Kruskal").apply_and_annotate() 
            # Not sure which stat test we should use, but in test_sig_diff() 
            # it mostly uses Kruskal, so I don't think it's a problem

            if plot:
                # Label and show
                ax.set_title(title)
                if "hue" in hue_plot_params:
                    plt.legend()

                fig_fname = pjoin(figpath, subj_id, title+'.jpg')    
                plt.savefig(fig_fname)
                # print('fig was saved to:', fig_fname)
                # plt.show()

            for st_res in annotator._get_results(num_comparisons=2):
                d = st_res.data
                pvalue = d.pvalue

#                 if len(var_name) == 0:
#                     g1 = '-'.join(d.group1)
#                     g2 = '-'.join(d.group2)
#                 else: # TODO
                g1 = d.group1
                g2 = d.group2

                res_name = f'{var_name}|{g1}-{g2}:pvalue'
                results[res_name] = pvalue
                
        return results


# -

def compare_between_vars_groupby_labels(
        subj_id,
        var1,
        var1_name,
        var2,
        var2_name,
        labels,
        labels_name,
        figpath,
        plot,
    ):
    
    if var1_name == labels_name or var2_name == labels_name:
        raise Exception('''Cannot compare between values of variable and labels (that were also used to group).
        (It does not make sense to compute this.)''')
        
    df = pd.DataFrame({
        var1_name: var1,
        var2_name: var2,
        labels_name: labels,
    })
    
    df_melt = pd.melt(df[[var1_name, var2_name]])
    df_melt[labels_name] = list(df[labels_name].values) * 2
    l_order = [l for l in labels_order[labels_name] if l in labels]
    pairs = [ [(s, var1_name), (s, var2_name)] for s in l_order if s in l_order]

    hue_plot_params = {
        "data": df_melt,
        "x": labels_name,
        "y": "value",
        "hue": "variable",
        "order": l_order,
    }
    # print(figpath)
    

    stats_result = plot_stats_test(
        hue_plot_params, pairs, plot,
        subj_id,
        f'{subj_id} | {var1_name} v.s. {var2_name}\ngroup_by: {labels_name}',
        fig_width[labels_name], 
        figpath,
        f'{var1_name}-{var2_name}',
    )
    
    return stats_result


def compare_one_var_groupby_labels(
        subj_id,
        var,
        var_name,
        labels,
        labels_name,
        figpath,
        plot,
    ):
    
    if var_name == labels_name:
        raise Exception('''Cannot compare between values of variable and labels (that were also used to group).
        (It does not make sense to compute this.)''')
        
    l_order = labels_order[labels_name]
    df = pd.DataFrame({
        var_name: var,
        labels_name: labels,
    })
    existing_label = np.unique(labels)
    
    l_order = [l for l in labels_order[labels_name] if l in labels]
    pairs = [(a, b) for (a, b) in labels_combination[labels_name] 
             if a in l_order and b in l_order]

    hue_plot_params = {
        "data": df,
        "x": labels_name,
        "y": var_name,
        "order": l_order,
    }
    
    stats_result = plot_stats_test(
        hue_plot_params, pairs, plot,
        subj_id,
        f'{subj_id} | {var_name}\ngroup_by: {labels_name}',
        fig_width[labels_name],
        figpath,
        var_name,
    )
    
    return stats_result


def assert_n_rows(all_subj_path, average_orp_every, average_irasa_every):
    # check whether all the csv files contain the same number of rows
    
    for subj_id in os.listdir(all_subj_path):
        
        if os.path.isdir(subj_id):
            subj_path = pjoin(all_subj_path, subj_id)
            n_rows_subj_df = -1

            for each_file in os.listdir(subj_path):

                if each_file == 'corr.csv':
                    continue

                df = pd.read_csv(pjoin(subj_path, each_file))
                print(each_file)
                print(df.head())

                current_n_rows_subj_df = len(df)

                if 'down' in each_file:
                    if 'orp' in each_file:
                        current_n_rows_subj_df /= average_orp_every
                    else:
                        current_n_rows_subj_df /= average_irasa_every

                if n_rows_subj_df == -1:
                    n_rows_subj_df = current_n_rows_subj_df
                else:
                    assert n_rows_subj_df == current_n_rows_subj_df

    print('All correctly done.')

    

    

    

    

    
