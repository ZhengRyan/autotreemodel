#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: feature_selection_2_treemodel.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-26
'''

import warnings

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm
from xgboost import XGBClassifier

from .utils import get_ks

warnings.filterwarnings('ignore')

from .logger_utils import Logger

log = Logger(level='info', name=__name__).logger


class ShapSelectFeature:
    def __init__(self, estimator, linear=False, estimator_is_fit_final=False):
        self.estimator = estimator
        self.linear = linear
        self.weight = None
        self.estimator_is_fit_final = estimator_is_fit_final

    def fit(self, X, y, exclude=None):
        '''

        Args:
            X:
            y:
            exclude:

        Returns:

        '''
        if exclude is not None:
            X = X.drop(columns=exclude)
        if not self.estimator_is_fit_final:
            self.estimator.fit(X, y)
        if self.linear:
            explainer = shap.LinearExplainer(self.estimator, X)
        else:
            estimator = self.estimator.get_booster()
            temp = estimator.save_raw()[4:]
            estimator.save_raw = lambda: temp
            explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        shap_abs = np.abs(shap_values)
        shap_importance_list = shap_abs.mean(0)
        self.weight = pd.DataFrame(shap_importance_list, index=X.columns, columns=['weight'])
        return self.weight


def corr_select_feature(frame, by='auc', threshold=0.95, return_frame=False):
    '''

    Args:
        frame:
        by:
        threshold:
        return_frame:

    Returns:

    '''
    if not isinstance(by, (str, pd.Series)):

        if isinstance(by, pd.DataFrame):
            if by.shape[1] == 1:
                by = pd.Series(by.iloc[:, 0].values, index=by.index)
            else:
                by = pd.Series(by.iloc[:, 1].values, index=by.iloc[:, 0].values)
            # by = pd.Series(by.iloc[:, 1].values, index=frame.columns)
        else:
            by = pd.Series(by, index=frame.columns)

    # ?????????????????????
    by.sort_values(ascending=False, inplace=True)
    # print('????????????????????????', by)

    # df = frame.copy()

    by.index = by.index.astype(type(list(frame.columns)[0]))
    df_corr = frame[list(by.index)].fillna(-999).corr().abs()  # ??????
    # df_corr = frame[list(by.index)].corr().abs()

    ix, cn = np.where(np.triu(df_corr.values, 1) > threshold)

    del_all = []

    if len(ix):

        for i in df_corr:

            if i not in del_all:
                # ??????????????????????????????????????????????????????
                del_tmp = list(df_corr[i][(df_corr[i] > threshold) & (df_corr[i] != 1)].index)

                # ????????????????????????????????????????????????????????????
                if del_tmp:
                    by_tmp = by.loc[del_tmp]
                    del_l = list(by_tmp[by_tmp <= by.loc[i]].index)
                    del_all.extend(del_l)

    del_f = list(set(del_all))

    if return_frame:
        r = frame.drop(columns=del_f)
        return (del_f, r)

    return del_f


def psi(no_base, base, return_frame=False):
    '''
    psi??????
    Args:
        no_base:??????????????????
        base:???????????????
        return_frame:?????????????????????psi?????????

    Returns:
        float???Series
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


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x


def calc_psi(no_base, base):
    '''
    psi?????????????????????
    Args:
        no_base: ??????????????????
        base: ???????????????

    Returns:
        float???DataFrame
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


def feature_select(datasets, fea_names, target, feature_select_method='shap', method_threhold=0.001,
                   corr_threhold=0.8, psi_threhold=0.1, params={}):
    '''

    Args:
        datasets:
        fea_names:
        target:
        feature_select_method:
        method_threhold:
        corr_threhold:
        psi_threhold:
        params:

    Returns:

    '''
    dev_data = datasets['dev']
    nodev_data = datasets['nodev']

    params = {
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 200),
        'max_depth': params.get('max_depth', 3),
        'min_child_weight': params.get('min_child_weight', 5),
        'subsample': params.get('subsample', 0.7),
        'colsample_bytree': params.get('colsample_bytree', 0.9),
        'colsample_bylevel': params.get('colsample_bylevel', 0.7),
        'gamma': params.get('gamma', 7),
        'reg_alpha': params.get('reg_alpha', 10),
        'reg_lambda': params.get('reg_lambda', 10)
    }

    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(dev_data[fea_names], dev_data[target])

    if feature_select_method == 'shap':
        shap_model = ShapSelectFeature(estimator=xgb_clf, estimator_is_fit_final=True)
        fea_weight = shap_model.fit(dev_data[fea_names], dev_data[target])
        fea_weight.sort_values(by='weight', inplace=True)
        fea_weight = fea_weight[fea_weight['weight'] >= method_threhold]
        log.info('Shap??????: {}'.format(method_threhold))
        log.info('Shap?????????????????????: {}'.format(len(fea_names) - fea_weight.shape[0]))
        log.info('Shap?????????????????????: {}'.format(fea_weight.shape[0]))
        fea_names = list(fea_weight.index)
        log.info('*' * 50 + 'Shap????????????' + '*' * 50)


    elif feature_select_method == 'feature_importance':
        fea_weight = pd.DataFrame(list(xgb_clf.get_booster().get_score(importance_type='gain').items()),
                                  columns=['fea_names', 'weight']
                                  ).sort_values('weight').set_index('fea_names')
        fea_weight = fea_weight[fea_weight['weight'] >= method_threhold]
        log.info('feature_importance??????: {}'.format(method_threhold))
        log.info('feature_importance?????????????????????: {}'.format(len(fea_names) - fea_weight.shape[0]))
        fea_names = list(fea_weight.index)
        log.info('feature_importance?????????????????????: {}'.format(fea_names))
        log.info('*' * 50 + 'feature_importance????????????' + '*' * 50)

    if corr_threhold:
        del_fea_list = corr_select_feature(dev_data[fea_names], by=fea_weight, threshold=0.8)
        log.info('???????????????: {}'.format(corr_threhold))
        log.info('??????????????????????????????: {}'.format(len(del_fea_list)))
        fea_names = [i for i in fea_names if i not in del_fea_list]
        # fea_names = list(set(fea_names) - set(del_fea_list))
        log.info('??????????????????????????????: {}'.format(len(fea_names)))
        log.info('*' * 50 + '?????????????????????' + '*' * 50)

    if psi_threhold:
        psi_df = psi(dev_data[fea_names], nodev_data[fea_names]).sort_values(0)
        psi_df = psi_df.reset_index()
        psi_df = psi_df.rename(columns={'index': 'fea_names', 0: 'psi'})
        psi_list = psi_df[psi_df.psi < psi_threhold].fea_names.tolist()
        log.info('PSI??????: {}'.format(psi_threhold))
        log.info('PSI?????????????????????: {}'.format(len(fea_names) - len(psi_list)))
        fea_names = [i for i in fea_names if i in psi_list]
        # fea_names = list(set(fea_names) and set(psi_list))
        log.info('PSI?????????????????????: {}'.format(len(fea_names)))
        log.info('*' * 50 + 'PSI????????????' + '*' * 50)

    return fea_names


def stepwise_del_feature(datasets, fea_names, target, params={}):
    '''

    Args:
        datasets:
        fea_names:
        target:
        params:

    Returns:

    '''
    log.info("????????????????????????")
    dev_data = datasets['dev']
    nodev_data = datasets['nodev']
    stepwise_del_params = {
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 200),
        'max_depth': params.get('max_depth', 3),
        'min_child_weight': params.get('min_child_weight', 5),
        'subsample': params.get('subsample', 0.7),
        'colsample_bytree': params.get('colsample_bytree', 0.9),
        'colsample_bylevel': params.get('colsample_bylevel', 0.7),
        'gamma': params.get('gamma', 7),
        'reg_alpha': params.get('reg_alpha', 10),
        'reg_lambda': params.get('reg_lambda', 10)
    }

    xgb_clf = XGBClassifier(**stepwise_del_params)
    xgb_clf.fit(dev_data[fea_names], dev_data[target])

    pred_test = xgb_clf.predict_proba(nodev_data[fea_names])[:, 1]
    pred_train = xgb_clf.predict_proba(dev_data[fea_names])[:, 1]

    test_ks = get_ks(nodev_data[target], pred_test)
    train_ks = get_ks(dev_data[target], pred_train)
    log.info('test_ks is : {}'.format(test_ks))
    log.info('train_ks is : {}'.format(train_ks))

    train_number, oldks, del_list = 0, test_ks, list()
    log.info('train_number: {}, test_ks: {}'.format(train_number, test_ks))

    # while True:
    #     flag = True
    #     for fea_name in tqdm(fea_names):
    #         print('??????{}???????????????'.format(fea_name))
    #         names = [fea for fea in fea_names if fea_name != fea]
    #         print('??????names is???', names)
    #         xgb_clf.fit(dev_data[names], dev_data[target])
    #         train_number += 1
    #         pred_test = xgb_clf.predict_proba(nodev_data[names])[:, 1]
    #         test_ks = get_ks(nodev_data[target], pred_test)
    #         if test_ks >= oldks:
    #             oldks = test_ks
    #             flag = False
    #             del_list.append(fea_name)
    #             log.info(
    #                 '??????????????????????????? train_number: {}, test_ks: {} by feature: {}'.format(train_number, test_ks, fea_name))
    #             fea_names = names
    #     if flag:
    #         print('=====================???????????????==========')
    #         break
    #     log.info("???????????????????????? train_number: %s, test_ks: %s del_list: %s" % (train_number, oldks, del_list))
    #     print('oldks is ???',oldks)
    #     print('fea_names is : ',fea_names)

    for fea_name in tqdm(fea_names):
        names = [fea for fea in fea_names if fea_name != fea]
        xgb_clf.fit(dev_data[names], dev_data[target])
        train_number += 1
        pred_test = xgb_clf.predict_proba(nodev_data[names])[:, 1]
        test_ks = get_ks(nodev_data[target], pred_test)
        if test_ks >= oldks:
            oldks = test_ks
            del_list.append(fea_name)
            log.info(
                '??????????????????????????? train_number: {}, test_ks: {} by feature: {}'.format(train_number, test_ks, fea_name))
            fea_names = names
    log.info("???????????????????????? train_number: %s, test_ks: %s del_list: %s" % (train_number, oldks, del_list))

    ########################
    log.info('???????????????????????????: {}'.format(del_list))
    fea_names = [i for i in fea_names if i not in del_list]
    # fea_names = list(set(fea_names) - set(del_list))
    log.info('???????????????????????????: {}'.format(len(fea_names)))
    log.info('*' * 50 + '??????????????????' + '*' * 50)

    return del_list, fea_names
