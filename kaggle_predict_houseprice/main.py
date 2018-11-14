# ◆◆◆◆◆◆◆◆ 1 导入数据 ◆◆◆◆◆◆◆
#1.1 导入相关包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth',200)
pd.set_option('display.max_column',200)
pd.set_option('display.width', 2000)

#查看数据
df = pd.read_csv(r'D:\4.program\Machinelearning\kaggle_predict_houseprice\train.csv')
df.head()
# ◆◆◆◆◆◆◆◆ 2 特征工程 ◆◆◆◆◆◆◆
#2.1 特征整体情况描述
print(df.isnull().any())
print(df.describe())
#2.1 特征关系情况描述
k = 12
corrmat = df.corr()
# 热力图展示关系
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# plt.show()

# f, ax = plt.subplots(figsize=(6, 6))
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # 选出和SalePrice相关系数最大的10个特征（包含SalePrice）
# cm = np.corrcoef(df[cols].values.T)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
train_data = df[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', '1stFlrSF', 'FullBath', 'TotalBsmtSF', 'YearBuilt',
       'YearRemodAdd']]
#2.1 特征异常值处理
print(train_data.describe())
print(train_data.isnull().any())
# 没有缺失值
# 对年特征进行转换

train_data['YearBuilt'] = train_data['YearBuilt'].apply(lambda x: 2018 - int(x))
train_data['YearRemodAdd'] = train_data['YearRemodAdd'].apply(lambda x: 2018 - int(x))
# ◆◆◆◆◆◆◆ 4 响应预测模型训练和选择
#
# #4.1 准备数据
# #去预测变量 SalePrice
#
# # 准备数据集
from sklearn.model_selection import train_test_split #分离器函数
from sklearn import preprocessing
X = train_data[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF', 'FullBath',  'YearBuilt',
       'YearRemodAdd']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 数据标准化
ss_x = preprocessing.StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

ss_y = preprocessing.StandardScaler()
y_train = ss_y.fit_transform(y_train.values.reshape(-1,1))
y_test = ss_y.transform(y_test.values.reshape(-1,1))


# #4.2 对比不同的分类算法

import pandas as pd
import numpy as np
# KNN回归
from sklearn.neighbors import KNeighborsRegressor
# 线性回归
from sklearn.linear_model import LinearRegression
# SVM回归
from sklearn.svm import SVR
# 决策树回归
from sklearn.tree import DecisionTreeRegressor
# 组合模型
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
from sklearn.neural_network import MLPRegressor
# 回归指标
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平均绝对误差 MAE
from sklearn.metrics import explained_variance_score #解释方差 MSLE
from sklearn.metrics import r2_score #确定系数


# #循环对回归个模型进行训练
classifiers = [
    KNeighborsRegressor(n_neighbors = 5),
    LinearRegression(),
    SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1),
    BaggingRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    RandomForestRegressor(),
    MLPRegressor(),
    DecisionTreeRegressor()]

# #计划用 pandas 的 dataframe 存放每一个模型的 评估值，结果保存在 result 中
cols = ['Classifier','mean_squared_error','mean_absolute_error','explained_variance_score'
    ,'r2_score']
result = pd.DataFrame(columns = cols)
#
# #对10个回归模型进行循环处理，依次输出他们的 评估值，结果保存在 result 中
for clf in classifiers:
    clf.fit(X_train,y_train)
    name = clf.__class__.__name__

    print('='*30)
    print(name)

    print('*******Result**********')
    y_pred = clf.predict(X_test)
    MSE = mean_squared_error(y_test,y_pred)
    print("MSE: %.4f" % MSE)

    print('*******Result**********')
    y_pred = clf.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    print('MAE:' + str(MAE))

    print('*******Result**********')
    y_pred = clf.predict(X_test)
    MSLE = explained_variance_score(y_test, y_pred)
    print('MSLE:' + str(MSLE))

    print('*******Result**********')
    y_pred = clf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print('r2_score:' + str(r2))

    #把当前循环分类器结果保存在 result_clf 的dataframe中
    # result_clf = pd.DataFrame([[name,MSE,MAE,MSLE,r2]])
    result_clf = pd.Series({'Classifier': name, 'mean_squared_error': MSE, 'mean_absolute_error':MAE,'explained_variance_score':MSLE,'r2_score':r2})
    #把 result_clf 合并在 result 中
    result = result.append(result_clf, ignore_index=True)

print('='*30)
print(result)
np.savetxt("各种回归模型结.txt",result,newline='\n',fmt='%s')

# #评估选择最好的模型
#
#选择最好的算法 MLPRegressor 重新训练模型
favorite_clf = MLPRegressor()
favorite_clf.fit(X_train,y_train)


# #调参  hyperopt
import hyperopt as hp
import sklearn.losses as losses
from hyperopt import fmin, tpe
space = {'window': hp.choice('window',[30, 60, 120, 180]),
        'units1': hp.choice('units1', [64, 512]),
        'units2': hp.choice('units2', [64, 512]),
        'units3': hp.choice('units3', [64, 512]),
        'lr': hp.choice('lr',[0.01, 0.001, 0.0001]),
        'activation': hp.choice('activation',['relu',
                                                'sigmoid',
                                                'tanh',
                                                'linear']),
        'solver' : hp.choice('solver',['lbfgs', 'sgd', 'adam']),
        'loss': hp.choice('loss', [losses.logcosh,
                                    losses.mse,
                                    losses.mae,
                                    losses.mape])}
#
# ◆ 5 保存模型
#     #有两种持久化的方法：
#     # 通过pickle.dump() 方法把模型保存为文件
#     # 通过pickle.load() 方法把模型读取（加载）为文件
#
# import pickle
# with open('response_model.pickle','wb') as fw:
#     pickle.dump(favorite_clf,fw)
# #调用系统命名 ls 查看当前目录下的文件
