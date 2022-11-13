#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

'''
@File: ryan_tutorial_code.py
@Author: RyanZheng
@Email: ryan.zhengrp@gmail.com
@Created Time on: 2020-04-27
'''

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from autotreemodel import AutoBuildTreeModel

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
ml_res_save_path = './examples/example_model_result/{}'.format(
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
#     './examples/example_data/TT01p1_id_y_fea_to_model.csv')
# ###TODO 注意修改，读取建模数据
#
# ###TODO 注意修改
# client_batch = 'TT01p1'
# key, target, data_type = 'id', 'target', 'type'  # key是主键字段名，target是目标变量y的字段名，data_type是train、test数据集标识的字段名
# ml_res_save_path = './examples/example_model_result/{}'.format(
#     client_batch)
# ###TODO 注意修改
#
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
