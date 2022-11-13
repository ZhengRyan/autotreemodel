#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: plot_metrics.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2022-08-26
'''

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


def get_optimal_cutoff(fpr_recall, tpr_precision, threshold, is_f1=False):
    if is_f1:
        youdenJ_f1score = (2 * tpr_precision * fpr_recall) / (tpr_precision + fpr_recall)
    else:
        youdenJ_f1score = tpr_precision - fpr_recall
    point_index = np.argmax(youdenJ_f1score)
    optimal_threshold = threshold[point_index]
    point = [fpr_recall[point_index], tpr_precision[point_index]]
    return optimal_threshold, point


def plot_ks(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)  ###计算ks的值

    plt.figure(figsize=(6, 6))
    x = np.arange(len(thresholds)) / len(thresholds)
    plt.plot(x, tpr, lw=1)
    plt.plot(x, fpr, lw=1)
    plt.plot(x, tpr - fpr, lw=1, linestyle='--', label='KS curve (KS = %0.4f)' % ks)

    optimal_th, optimal_point = get_optimal_cutoff(fpr_recall=fpr, tpr_precision=tpr, threshold=thresholds)
    optimal_th_index = np.where(thresholds == optimal_th)
    plt.plot(optimal_th_index[0][0] / len(thresholds), ks, marker='o', color='r')
    plt.text(optimal_th_index[0][0] / len(thresholds), ks, (float('%.4f' % optimal_point[0]),
                                                            float('%.4f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_th_index[0][0] / len(thresholds), ks, f'threshold:{optimal_th:.4f}', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Thresholds Index')
    plt.ylabel('TPR FPR KS')
    name = '{} KS Curve'.format(title)
    plt.title(name)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(output_path + name, bbox_inches='tight')
    plt.show()
    return {'KS': ks, 'KS最大值-threshold': optimal_th}


def plot_roc(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_pred)  ###计算auc的值

    lw = 2
    plt.figure(figsize=(6, 6))

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.4f)' % auc_value)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    optimal_th, optimal_point = get_optimal_cutoff(fpr_recall=fpr, tpr_precision=tpr, threshold=thresholds)
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], (float('%.4f' % optimal_point[0]),
                                                  float('%.4f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_point[0], optimal_point[1], f'threshold:{optimal_th:.4f}', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    name = '{} ROC Curve'.format(title)
    plt.title(name)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(output_path + name, bbox_inches='tight')
    plt.show()
    return {'AUC': auc_value}


def plot_pr(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1score = (2 * precision * recall) / (precision + recall)  ###计算F1score
    max_f1score = max(f1score)

    lw = 2
    plt.figure(figsize=(6, 6))

    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (F1score = %0.4f)' % max_f1score)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    optimal_th, optimal_point = get_optimal_cutoff(fpr_recall=recall, tpr_precision=precision, threshold=thresholds,
                                                   is_f1=True)
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], (float('%.4f' % optimal_point[0]),
                                                  float('%.4f' % optimal_point[1])),
             ha='right', va='top', fontsize=12)
    plt.text(optimal_point[0], optimal_point[1], f'threshold:{optimal_th:.4f}', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    name = '{} PR Curve'.format(title)
    plt.title(name)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(output_path + name, bbox_inches='tight')
    plt.show()
    return {'F1_Score最大值': max_f1score, 'F1_Score最大值-threshold': optimal_th, '模型拐点': optimal_th, '阀值': optimal_th,
            'Precision': optimal_point[1], 'Recall': optimal_point[0], 'F1_Score': max_f1score}


def plot_pr_f1(y_true: pd.Series, y_pred: pd.Series, output_path='', title=''):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    thresholds = np.insert(thresholds, 0, 0, axis=None)
    f1score = (2 * precision * recall) / (precision + recall)  ###计算F1score

    x = np.arange(len(thresholds)) / len(thresholds)

    pr_f1_dict = {'Precision': precision, 'Recall': recall, 'F1_score': f1score}

    for i in pr_f1_dict:
        plt.figure(figsize=(6, 6))

        plt.plot(x, pr_f1_dict[i], lw=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Thresholds Index')
        plt.ylabel('{}'.format(i))
        name = '{} {} Curve'.format(title, i)
        plt.title(name)
        plt.savefig(output_path + name, bbox_inches='tight')
        plt.show()

    return {'Thresholds': list(thresholds), '模型召回率': list(recall), '模型精准率': list(precision), '模型F1-score': list(f1score)}


def calc_celue_cm(df: pd.DataFrame, target='target', to_bin_col='p'):
    q_cut_list = np.arange(0, 1, 1 / 10) + 0.1
    confusion_matrix_df = pd.DataFrame()
    for i in q_cut_list:
        df['pred_label'] = np.where(df[to_bin_col] >= i, 1, 0)
        tmp_list = []
        tmp_list.append(i)

        tn, fp, fn, tp = confusion_matrix(np.array(df[target]), np.array(df['pred_label'])).ravel()

        tmp_list.extend([tp, fp, tn, fn])

        confusion_matrix_df = confusion_matrix_df.append(pd.DataFrame(tmp_list).T)

    # confusion_matrix_df.columns = ['阈值', 'TP', 'FP', 'TN', 'FN']
    confusion_matrix_df.columns = ['阈值', '实际正样本-预测为正样本', '实际负样本-预测为正样本', '实际负样本-预测为负样本', '实际正样本-预测为负样本']
    confusion_matrix_df.set_index('阈值', inplace=True)
    confusion_matrix_df['sum'] = confusion_matrix_df.apply(lambda x: x.sum(), axis=1)

    # return confusion_matrix_df
    return confusion_matrix_df.to_dict()


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


def calc_plot_metrics(df: pd.DataFrame, to_bin_col='p', target='target', curve_save_path=''):
    data = {k: v for k, v in df.groupby('type')}
    data.update({'all': df})

    for data_type, type_df in data.items():
        res_save_path = curve_save_path + '/{}/'.format(data_type)
        os.makedirs(res_save_path, exist_ok=True)

        res_dict = {}
        res_dict.update(plot_roc(type_df[target], type_df[to_bin_col], res_save_path, data_type))
        res_dict.update(plot_ks(type_df[target], type_df[to_bin_col], res_save_path, data_type))
        res_dict.update(plot_pr(type_df[target], type_df[to_bin_col], res_save_path, data_type))
        res_dict.update(calc_celue_cm(type_df, target, to_bin_col))
        res_dict.update(plot_pr_f1(type_df[target], type_df[to_bin_col], res_save_path, data_type))

        ###相关指标保存json格式
        save_json(res_dict, res_save_path + '{}_res_json.json'.format(data_type))


if __name__ == "__main__":
    ######读取数据
    data_path = '/Users/ryanzheng/td/项目/RTA/还呗RTA/还呗建模环境账号信息/还呗建模环境/202208_事项/模型/TD47p25combine/TD47p25combine_td_to_report_data.csv'
    df = pd.read_csv(data_path)
    df = df[df['label'].notnull()]

    print(len(df))

    ######结果保存路径
    curve_save_path = '/Users/ryanzheng/PycharmProjects/auto_build_tree_model/examples/curve_result'
    calc_plot_metrics(df=df, to_bin_col='td', target='label', curve_save_path=curve_save_path)
