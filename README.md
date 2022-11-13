#自动构建树模型
##自动构建xgboost或lightgbm模型

### 一、训练、自动选变量、自动调参数
1、训练模型

2、shap 或 feature importance自动筛选变量

3、相关性筛选变量

4、PSI筛选变量

5、自动调参

6、逐步剔除变量

7、构建最终模型

### 二、建模相关结果保存
1、将模型文件持久化

2、将变量重要性持久化

3、将模型效果持久化

4、500条x数据用于验证后续部署是否一致

5、模型在建模数据集上的预测结果持久化

## 三、使用教程
请查看ryan_tutorial_code.py。里面有两个例子，一个列子使用的数据集随机生成的数据，一个是虚构现实数据


## 四、依赖包安装（建议先创建虚拟环境，不创建虚拟环境也行，创建虚拟环境是为了不和其它项目有依赖包的冲突，不创建虚拟环境的话在基础python环境执行pip install即可）
####创建虚拟环境
conda create -y --force -n auto_build_tree_model python=3.7.2
####激活虚拟环境
conda activate auto_build_tree_model

### 依赖包安装方式一

####安装依赖包
pip install pandas==1.2.4

pip install joblib==0.14.1

pip install xgboost==1.2.0

pip install bayesian-optimization==1.1.0

pip install lightgbm==3.2.1

pip install shap==0.36.0

### 依赖包安装方式二，执行如下命令安装依赖的包
pip install -r requirements.txt
