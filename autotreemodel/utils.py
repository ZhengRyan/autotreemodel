#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: utils.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-20
'''

import json
import math
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, r2_score, recall_score, precision_score, f1_score

FILLNA = -999


# auc
def get_auc(target, y_pred):
    if target.nunique() != 2:
        raise ValueError('the target is not 2 classier target')
    else:
        return roc_auc_score(target, y_pred)


# ks
def get_ks(target, y_pred):
    df = pd.DataFrame({
        'y_pred': y_pred,
        'target': target,
    })
    crossfreq = pd.crosstab(df['y_pred'], df['target'])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['ks'] = abs(crossdens[0] - crossdens[1])
    ks = max(crossdens['ks'])
    return ks


def to_score(x, A=404.65547022, B=72.1347520444):
    if x <= 0.001:
        x = 0.001
    elif x >= 0.999:
        x = 0.999

    result = round(A - B * math.log(x / (1 - x)))

    if result < 0:
        result = 0
    if result > 1200:
        result = 1200
    result = 1200 - result
    return result


def filter_miss(df, miss_threshold=0.9):
    '''

    :param df: 数据集
    :param miss_threshold: 缺失率大于等于该阈值的变量剔除
    :return:
    '''
    names_list = []
    for name, series in df.items():
        n = series.isnull().sum()
        miss_q = n / series.size
        if miss_q < miss_threshold:
            names_list.append(name)
    return names_list


def dump_model_to_file(model, path):
    pickle.dump(model, open(path, "wb"))


def load_model_from_file(path):
    return pickle.load(open(path, 'rb'))


def read_sql_string_from_file(path):
    with open(path, 'r', encoding='utf-8') as fb:
        sql = fb.read()
        return sql


def save_json(res_dict, file, indent=4):
    if isinstance(file, str):
        of = open(file, 'w')

    with of as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=indent)


def load_json(file):
    if isinstance(file, str):
        of = open(file, 'r')

    with of as f:
        res_dict = json.load(f)

    return res_dict


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def psi(no_base, base, return_frame=False):
    '''
    psi计算
    :param no_base: 非基准数据集
    :param base: 基准数据集
    :param return_frame: 是否返回详细的psi数据集
    :return:
    '''
    psi = list()
    frame = list()

    if isinstance(no_base, pd.DataFrame):
        for col in no_base:
            p, f = calc_psi(no_base[col], base[col])
            psi.append(p)
            frame.append(f)

        psi = pd.Series(psi, index=no_base.columns)

        frame = pd.concat(
            frame,
            keys=no_base.columns,
            names=['columns', 'id'],
        ).reset_index()
        frame = frame.drop(columns='id')
    else:
        psi, frame = calc_psi(no_base, base)

    res = (psi,)

    if return_frame:
        res += (frame,)

    return unpack_tuple(res)


def calc_psi(no_base, base):
    '''
    psi计算的具体逻辑
    :param no_base: 非基准数据集
    :param base: 基准数据集
    :return:
    '''
    no_base_prop = pd.Series(no_base).value_counts(normalize=True, dropna=False)
    base_prop = pd.Series(base).value_counts(normalize=True, dropna=False)

    psi = np.sum((no_base_prop - base_prop) * np.log(no_base_prop / base_prop))

    frame = pd.DataFrame({
        'no_base': no_base_prop,
        'base': base_prop,
    })
    frame.index.name = 'value'

    return psi, frame.reset_index()


# corr
def get_corr(df):
    return df.corr()


# accuracy
def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


# precision
def get_precision(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return precision_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# recall
def get_recall(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return recall_score(y_true, y_pred)
    else:
        return precision_score(y_true, y_pred, average='macro')


# f1
def get_f1(y_true, y_pred):
    if len(np.unique(y_true)) == 2:
        return f1_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average='macro')


# r2
def r2(preds, target):
    return r2_score(target, preds)


# vif
def get_vif(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    vif = 1 / (1 - r2)
    return vif


def get_best_threshold(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    ks = list(tpr - fpr)
    thresh = threshold[ks.index(max(ks))]
    return thresh


def get_bad_rate(df):
    return df.sum() / df.count()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
