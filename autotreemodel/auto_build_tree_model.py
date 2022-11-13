#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: auto_build_tree_model.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-27
'''

import gc
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .bayes_opt_tuner import classifiers_model_auto_tune_params
from .feature_selection_2_treemodel import feature_select, stepwise_del_feature, psi
from .utils import get_ks, get_auc, to_score

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class AutoBuildTreeModel():
    def __init__(self, datasets, fea_names, target, key='key', data_type='type',
                 no_feature_names=['key', 'target', 'apply_time', 'type'], ml_res_save_path='./model_result',
                 to_score_a_b={'A': 404.65547022, 'B': 72.1347520444}):

        if data_type not in datasets:
            raise KeyError('train、test数据集标识的字段名不存在！或未进行数据集的划分，请将数据集划分为train、test！！！')

        data_type_ar = np.unique(datasets[data_type])
        if 'train' not in data_type_ar:
            raise KeyError("""没有开发样本，数据集标识字段{}没有`train`该取值！！！""".format(data_type))

        if 'test' not in data_type_ar:
            raise KeyError("""没有验证样本，数据集标识字段{}没有`test`该取值！！！""".format(data_type))

        if target not in datasets:
            raise KeyError('样本中没有目标变量y值！！！')

        # fea_names = [i for i in fea_names if i != key and i != target]
        fea_names = [i for i in fea_names if i not in no_feature_names]
        log.info('数据集变量个数 : {}'.format(len(fea_names)))
        log.info('fea_names is : {}'.format(fea_names))

        self.datasets = datasets
        self.fea_names = fea_names
        self.target = target
        self.key = key
        self.no_feature_names = no_feature_names
        self.ml_res_save_path = ml_res_save_path + '/' + time.strftime('%Y%m%d%H%M%S_%S', time.localtime())
        self.to_score_a_b = to_score_a_b
        self.min_child_samples = max(round(len(datasets[datasets['type'] == 'train']) * 0.02),
                                     50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合
        # self.min_child_samples = max(round(len(datasets['dev']) * 0.02), 50)  # 一个叶子上数据的最小数量. 可以用来处理过拟合

        os.makedirs(self.ml_res_save_path, exist_ok=True)

    def fit(self, is_feature_select=True, is_auto_tune_params=True, is_stepwise_del_feature=True,
            feature_select_method='shap', method_threhold=0.001,
            corr_threhold=0.8, psi_threhold=0.2):
        '''

        Args:
            is_feature_select:
            is_auto_tune_params:
            feature_select_method:
            method_threhold:
            corr_threhold:
            psi_threhold:

        Returns: xgboost.sklearn.XGBClassifier或lightgbm.sklearn.LGBClassifier；list
            返回最优模型，入模变量list

        '''
        log.info('*' * 30 + '开始自动建模' + '*' * 30)

        log.info('*' * 30 + '获取变量名和数据集' + '*' * 30)
        fea_names = self.fea_names.copy()
        dev_data = self.datasets[self.datasets['type'] == 'train']
        nodev_data = self.datasets[self.datasets['type'] == 'test']

        del self.datasets;
        gc.collect()

        # dev_data = self.datasets['dev']
        # nodev_data = self.datasets['nodev']

        # params = {
        #     'learning_rate': 0.05,
        #     'n_estimators': 200,
        #     'max_depth': 3,
        #     'min_child_weight': 5,
        #     'gamma': 7,
        #     'subsample': 0.7,
        #     'colsample_bytree': 0.9,
        #     'colsample_bylevel': 0.7,
        #     'reg_alpha': 10,
        #     'reg_lambda': 10,
        #     'scale_pos_weight': 1
        # }
        # log.info('默认参数 {}'.format(params))
        #
        # log.info('构建基础模型')

        if is_feature_select:
            log.info('需要进行变量筛选')
            fea_names = feature_select({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target,
                                       feature_select_method, method_threhold,
                                       corr_threhold,
                                       psi_threhold)

        if is_auto_tune_params:
            log.info('需要进行自动调参')
            best_model = classifiers_model_auto_tune_params(train_data=(dev_data[fea_names], dev_data[self.target]),
                                                            test_data=(nodev_data[fea_names], nodev_data[self.target]))
            params = best_model.get_params()

        if is_stepwise_del_feature:
            log.info('需要逐步的删除变量')
            _, fea_names = stepwise_del_feature({'dev': dev_data, 'nodev': nodev_data}, fea_names, self.target, params)

        # 最终模型
        log.info('使用自动调参选出来的最优参数+筛选出来的变量，构建最终模型')
        log.info('最终变量的个数{}, 最终变量{}'.format(len(fea_names), fea_names))
        log.info('自动调参选出来的最优参数{}'.format(params))
        xgb_clf = XGBClassifier(**params)
        xgb_clf.fit(dev_data[fea_names], dev_data[self.target])

        # ###
        # pred_nodev = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        # pred_dev = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        # df_pred_nodev = pd.DataFrame({'target': nodev_data[self.target], 'p': pred_nodev}, index=nodev_data.index)
        # df_pred_dev = pd.DataFrame({'target': dev_data[self.target], 'p': pred_dev}, index=dev_data.index)
        # ###

        # ###
        # df_pred_nodev = nodev_data[self.no_feature_names + fea_names]
        # df_pred_dev = dev_data[self.no_feature_names + fea_names]
        # df_pred_nodev['p'] = xgb_clf.predict_proba(df_pred_nodev[fea_names])[:, 1]
        # df_pred_dev['p'] = xgb_clf.predict_proba(df_pred_dev[fea_names])[:, 1]
        # ###

        ###
        df_pred_nodev = nodev_data[self.no_feature_names]
        df_pred_dev = dev_data[self.no_feature_names]
        df_pred_nodev['p'] = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
        df_pred_dev['p'] = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]
        ###

        # 计算auc、ks、psi
        test_ks = get_ks(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_ks = get_ks(df_pred_dev[self.target], df_pred_dev['p'])
        test_auc = get_auc(df_pred_nodev[self.target], df_pred_nodev['p'])
        train_auc = get_auc(df_pred_dev[self.target], df_pred_dev['p'])

        q_cut_list = np.arange(0, 1, 1 / 20)
        bins = np.append(np.unique(np.quantile(df_pred_nodev['p'], q_cut_list)), df_pred_nodev['p'].max() + 0.1)
        df_pred_nodev['range'] = pd.cut(df_pred_nodev['p'], bins=bins, precision=0, right=False).astype(str)
        df_pred_dev['range'] = pd.cut(df_pred_dev['p'], bins=bins, precision=0, right=False).astype(str)
        nodev_psi = psi(df_pred_nodev['range'], df_pred_dev['range'])
        res_dict = {'dev_auc': train_auc, 'nodev_auc': test_auc, 'dev_ks': train_ks, 'nodev_ks': test_ks,
                    'nodev_dev_psi': nodev_psi}
        log.info('auc & ks & psi: {}'.format(res_dict))
        log.info('*' * 30 + '自动构建模型完成！！！' + '*' * 30)

        ##############
        log.info('*' * 30 + '建模相关结果开始保存！！！' + '*' * 30)
        joblib.dump(xgb_clf.get_booster(), '{}/xgb.ml'.format(self.ml_res_save_path))
        joblib.dump(xgb_clf, '{}/xgb_sk.ml'.format(self.ml_res_save_path))
        json.dump(xgb_clf.get_params(), open('{}/xgb.params'.format(self.ml_res_save_path), 'w'))
        xgb_clf.get_booster().dump_model('{}/xgb.txt'.format(self.ml_res_save_path))
        pd.DataFrame([res_dict]).to_csv('{}/xgb_auc_ks_psi.csv'.format(self.ml_res_save_path), index=False)

        pd.DataFrame(list(xgb_clf.get_booster().get_fscore().items()),
                     columns=['fea_names', 'weight']
                     ).sort_values('weight', ascending=False).set_index('fea_names').to_csv(
            '{}/xgb_featureimportance.csv'.format(self.ml_res_save_path))

        nodev_data[self.no_feature_names + fea_names].head(500).to_csv('{}/xgb_input.csv'.format(self.ml_res_save_path),
                                                                       index=False)

        ##############pred to score
        df_pred_nodev['score'] = df_pred_nodev['p'].map(
            lambda x: to_score(x, self.to_score_a_b['A'], self.to_score_a_b['B']))
        df_pred_dev['score'] = df_pred_dev['p'].map(
            lambda x: to_score(x, self.to_score_a_b['A'], self.to_score_a_b['B']))
        ##############pred to score

        df_pred_nodev.append(df_pred_dev).to_csv('{}/xgb_pred_to_report_data.csv'.format(self.ml_res_save_path),
                                                 index=False)

        log.info('*' * 30 + '建模相关结果保存完成！！！保存路径为：{}'.format(self.ml_res_save_path) + '*' * 30)

        return xgb_clf, fea_names


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************
    X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=328)
    data = pd.DataFrame(X)
    data['target'] = y
    data['key'] = [i for i in range(len(data))]
    data.columns = ['f0_radius', 'f0_texture', 'f0_perimeter', 'f0_area', 'f0_smoothness',
                    'f0_compactness', 'f0_concavity', 'f0_concave_points', 'f0_symmetry',
                    'f0_fractal_dimension', 'f1_radius_error', 'f1_texture_error', 'f1_perimeter_error',
                    'f2_area_error', 'f2_smoothness_error', 'f2_compactness_error', 'f2_concavity_error',
                    'f2_concave_points_error', 'f2_symmetry_error', 'f2_fractal_dimension_error',
                    'f3_radius', 'f3_texture', 'f3_perimeter', 'f3_area', 'f3_smoothness',
                    'f3_compactness', 'f3_concavity', 'f3_concave_points', 'f3_symmetry',
                    'f3_fractal_dimension', 'target', 'key']

    dev, nodev = train_test_split(data, test_size=0.3, random_state=328)
    dev['type'] = 'train'
    nodev['type'] = 'test'
    data = dev.append(nodev)

    ###TODO 注意修改
    client_batch = 'TT00p1'
    key, target, data_type = 'key', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
    ml_res_save_path = '../examples/example_model_result/{}'.format(
        client_batch)
    ###TODO 注意修改

    ###TODO 下面代码基本可以不用动
    # 初始化
    autobtmodel = AutoBuildTreeModel(datasets=data,  # 训练模型的数据集
                                     fea_names=list(data.columns),  # 数据集的字段名
                                     target=target,  # 目标变量y字段名
                                     key=key,  # 主键字段名
                                     data_type=data_type,  # train、test数据集标识的字段名
                                     no_feature_names=[key, target, data_type],  # 数据集中不用于开发模型的特征字段名，即除了x特征的其它字段名
                                     ml_res_save_path=ml_res_save_path,  # 建模相关结果保存路径
                                     )

    # 训练模型
    model, in_model_fea = autobtmodel.fit(is_feature_select=True,  # 特征筛选
                                          is_auto_tune_params=True,  # 是否自动调参
                                          is_stepwise_del_feature=True,  # 是进行逐步的变量删除
                                          feature_select_method='shap',  # 特征筛选指标
                                          method_threhold=0.001,  # 特征筛选指标阈值
                                          corr_threhold=0.8,  # 相关系数阈值
                                          psi_threhold=0.1,  # PSI阈值
                                          )
    ###TODO 上面代码基本可以不用动
    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************
    ##**************************************************随机生成的数据例子**************************************************

    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
    #
    # ###TODO 注意修改，读取建模数据
    # data = pd.read_csv(
    #     '../examples/example_data/TT01p1_id_y_fea_to_model.csv')
    # ###TODO 注意修改，读取建模数据
    #
    # ###TODO 注意修改
    # client_batch = 'TT01p1'
    # key, target, data_type = 'id', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
    # ml_res_save_path = '../examples/example_model_result/{}'.format(
    #     client_batch)
    # ###TODO 注意修改
    #
    # ###TODO 下面代码基本可以不用动
    # # 初始化
    # autobtmodel = AutoBuildTreeModel(datasets=data,  # 训练模型的数据集
    #                                  fea_names=list(data.columns),  # 数据集的字段名
    #                                  target=target,  # 目标变量y字段名
    #                                  key=key,  # 主键字段名
    #                                  data_type=data_type,  # train、test数据集标识的字段名
    #                                  no_feature_names=[key, target, data_type] + ['apply_time'],
    #                                  # 数据集中不用于开发模型的特征字段名，即除了x特征的其它字段名
    #                                  ml_res_save_path=ml_res_save_path,  # 建模相关结果保存路径
    #                                  )
    #
    # # 训练模型
    # model, in_model_fea = autobtmodel.fit(is_feature_select=True,  # 特征筛选
    #                                       is_auto_tune_params=True,  # 是否自动调参
    #                                       is_stepwise_del_feature=True,  # 是进行逐步的变量删除
    #                                       feature_select_method='shap',  # 特征筛选指标
    #                                       method_threhold=0.001,  # 特征筛选指标阈值
    #                                       corr_threhold=0.8,  # 相关系数阈值
    #                                       psi_threhold=0.1,  # PSI阈值
    #                                       )
    # ###TODO 上面代码基本可以不用动
    #
    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
    # ##**************************************************虚构现实数据例子**************************************************
