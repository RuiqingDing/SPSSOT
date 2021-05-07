import numpy as np
import pandas as pd
import random

def cal_min_max(df, feas):
    '''
    calculate features' mean and std for normalization
    :param df: data
    :param feas: feature name list
    :return: two array (mean and std)
    '''
    mins = []
    maxs = []
    for fea in feas:
        mins.append(df[fea].min())
        maxs.append(df[fea].max())
    return np.array(mins), np.array(maxs)

def normalize_maxmin(df, feas):
    for fea in feas:
        df[fea] = df[fea].fillna(df[fea].mean())
        df[fea] = df[fea].fillna(df[fea].mean())

    mins_s, maxs_s = cal_min_max(df, feas)
    X = (df[feas].values - mins_s) / (maxs_s - mins_s)
    X[np.isnan(X)] = 0
    y = df['label'].values

    return X, y

def precess_data(df_source, df_target_label, df_target_unlabel, feas):
    '''minmax'''
    Xs, ys = normalize_maxmin(df_source, feas)
    Xtl, ytl = normalize_maxmin(df_target_label, feas)
    Xtu, ytu = normalize_maxmin(df_target_unlabel, feas)

    return Xs, ys, Xtl, ytl, Xtu, ytu


def load_data(source, target, percent, source_percent=1.0):
    df_source1 = pd.read_csv(f'data/{source}/{percent}/{source}_label.csv')
    df_source2 = pd.read_csv(f'data/{source}/{percent}/{source}_unlabel.csv')
    df_source3 = pd.read_csv(f'data/{source}/{source}_test.csv')
    df_source = pd.concat([df_source1, df_source2, df_source3], ignore_index=True)
    
    if source_percent < 1.0:
        grouped = df_source.groupby('PATIENT_ID')['label'].sum().reset_index()
        pids_sepsis = grouped[grouped.label > 0]['PATIENT_ID'].tolist()
        pids_no = grouped[grouped.label == 0]['PATIENT_ID'].tolist()
        
        random.seed(42)
        pids_sample_sepsis = random.sample(pids_sepsis, int(source_percent*len(pids_sepsis)))
        random.seed(42)
        pids_sample_no = random.sample(pids_no, int(source_percent*len(pids_no)))
        pids_sample = pids_sample_sepsis + pids_sample_no
    
        df_source = df_source[df_source.PATIENT_ID.isin(pids_sample)]
        
    df_source = df_source.sample(frac=1.0, random_state=42)

    df_target_unlabel = pd.read_csv(f'data/{target}/{percent}/{target}_unlabel.csv')
    df_target_label = pd.read_csv(f'data/{target}/{percent}/{target}_label.csv')

    # shuffle data
    df_target_label = df_target_label.sample(frac=1.0, random_state=42)
    df_target_unlabel = df_target_unlabel.sample(frac=1.0, random_state=42)

    feas = list(set(df_source.columns) - set(['PATIENT_ID', 'label']))
    Xs, ys, Xtl, ytl, Xtu, ytu = precess_data(df_source, df_target_label, df_target_unlabel, feas)
    return Xs, ys, Xtl, ytl, Xtu, ytu

def load_test_data(target):
    df_test_label = pd.read_csv(f'data/{target}/{target}_test.csv')
    feas = list(set(df_test_label.columns) - set(['PATIENT_ID', 'label']))
    X, y = normalize_maxmin(df_test_label, feas)
    return X, y
