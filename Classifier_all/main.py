# -*- coding: UTF-8 -*- 
# @Author: Iris_biubiu
# @Time  : 2018-11
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
def data_explore(data):
    explore_data = data
    return explore_data
def data_process(data):
    processed_data = data
    return processed_data
# 选择模型
def classifier_select(x_train,x_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc

    classifier = [
        KNeighborsClassifier(n_neighbors=5),
        SVC(kernel='rbf', C=0.01, probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        XGBClassifier()
    ]
    cols = ['classifier']
    result = pd.DataFrame(columns=cols)
    for clf in classifier:
        name = clf.__class__.__name__
        clf.fit(x_train,y_train)
        # 对测试集预测
        print("☆"*15+'-'*4+name+'-'*4+"☆"*15)
        y_pred = clf.predict(x_test)
        y_pred_proda = clf.predict_proba(x_test)[:, 1]

        acc = accuracy_score(y_test,y_pred)
        print("acc" + str(acc))

        # auc_value = roc_auc_score(y_test,y_pred_proda)
        # print("auc_value:" + str(auc_value))
        auc_value = ''

        result_clf = pd.Series({"classifier": name, 'acc': acc ,'auc_value': auc_value})
        result = result.append(result_clf, ignore_index=True)
        np.savetxt("分类结果_" + name + ".txt", pd.DataFrame(y_pred), fmt='%s', newline='/n')
# 利用GridSearchCV对模型进行调参
def classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid):
    clf = clf
    from sklearn.model_selection import GridSearchCV
    optimized_clf = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1,
                                 n_jobs=-1)
    optimized_clf.fit(x_pre_data, y_pre_data)
    print("★" * 15 + '-' * 4 + clf.__class__.__name__ + '-' * 4 + "★" * 15)
    cv_result = pd.DataFrame.from_dict(optimized_clf.cv_results_)
    with open('class_' + clf.__class__.__name__ + '_result.csv', 'w') as f:
        cv_result.to_csv(f)
    print('♣参数的最佳取值：{0}'.format(optimized_clf.best_params_))
    print('♣最佳模型得分:{0}'.format(optimized_clf.best_score_))

# knn模型训练
def knn_train(x_pre_data, y_pre_data):
    from sklearn.neighbors import KNeighborsClassifier
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 11)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]
    clf = KNeighborsClassifier()
    classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid)
# svm模型训练
def svm_train(x_pre_data, y_pre_data):
    from sklearn.svm import SVC
    param_grid = {'kernel': ('linear', 'rbf'), 'C': range(1, 10, 1), 'gamma': np.arange(0, 10, 0.1)}
    clf = SVC()
    classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid)
# DecisionTree模型训练
def decisiontree_train(x_pre_data, y_pre_data):
    from sklearn.tree import DecisionTreeClassifier
    param_grid = {'max_depth': range(1, 10, 1),
                  'max_features': ['auto', 'sqrt', 'log2']}
    clf = DecisionTreeClassifier()
    classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid)
# randomforest模型训练
def randomforest_train(x_pre_data, y_pre_data):
    from sklearn.ensemble import RandomForestClassifier
    param_grid = {'max_features': ['auto', 'sqrt', 'log2'], 'n_estimators': range(10, 20, 1),
                  'min_samples_leaf': range(1, 500, 50)}
    clf = RandomForestClassifier()
    classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid)
# adaboost模型训练
def adaboost_train(x_pre_data, y_pre_data):
    from sklearn.ensemble import AdaBoostClassifier
    param_grid = {'n_estimators': range(1, 100, 10),
                  'learning_rate': np.arange(0.1, 1, 0.1)}
    clf = AdaBoostClassifier()
    classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid)
# GDBT模型训练
def GDBT_train(x_pre_data, y_pre_data):
    from sklearn.ensemble import GradientBoostingClassifier
    param_grid = {'n_estimators': range(1, 100, 10), 'subsample': np.arange(0.1, 1, 0.1),
                  'min_samples_leaf': range(1, 500, 50),'max_depth': range(1, 10, 1),
                  'max_features': ['auto', 'sqrt', 'log2']}
    clf = GradientBoostingClassifier()
    classifier_train_gsc(x_pre_data, y_pre_data, clf, param_grid)
# xgboost模型训练
def xgboost_train(x_pre_data, y_pre_data):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    # 已确定的参数 'n_estimators': 30,'max_depth': 3, 'min_child_weight': 1,'gamma': 0.01,'subsample': 0.7, 'colsample_bytree': 0.5,'reg_alpha': 0, 'reg_lambda': 0,'learning_rate': 0.07
    # cv_params = {'n_estimators': [20, 25, 30, 35]}
    # cv_params = {'max_depth': [i for i in range(1, 11)],'min_child_weight':[i for i in range(1, 11)]}
    # cv_params = {'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]}
    # cv_params = {'colsample_bytree': [0.3, 0.4, 0.5, 0.6], 'subsample': [0.6, 0.7, 0.8, 0.9]}
    # cv_params = {'reg_alpha': [0, 0.01, 0.05, 0.75, 1], 'reg_lambda': [0, 0.01, 0.05, 0.75, 1]}
    cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    other_params = {'learning_rate': 0.07, 'n_estimators': 30, 'max_depth': 3, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.5, 'gamma': 0.01, 'reg_alpha': 0, 'reg_lambda': 0}
    clf = XGBClassifier(**other_params)
    optimized_clf = GridSearchCV(estimator=clf, param_grid=cv_params, scoring='accuracy', cv=5, verbose=1, n_jobs=4)
    optimized_clf.fit(x_pre_data, y_pre_data)
    print("☆" * 15 + '-' * 4 + clf.__class__.__name__ + '-' * 4 + "☆" * 15)
    cv_result = pd.DataFrame.from_dict(optimized_clf.cv_results_)
    with open('class_' + clf.__class__.__name__ + '_result.csv', 'w') as f:
        cv_result.to_csv(f)
    print('参数的最佳取值：{0}'.format(optimized_clf.best_params_))
    print('最佳模型得分:{0}'.format(optimized_clf.best_score_))



if __name__ == "__main__":
    from sklearn.datasets import load_iris
    data = load_iris()
    '''
    数据探查
    '''
    data_explore(data)
    '''
    特征工程
    '''
    features_goal = 'Survived'
    pre_data = data_process(data)
    '''
    模型选择
    '''
    # 准备数据集
    from sklearn.model_selection import train_test_split

    # x_pre_data = pre_data.drop(features_goal, axis=1)
    # y_pre_data = pre_data[features_goal]
    x_pre_data = data.data
    y_pre_data = data.target
    x_train, x_test, y_train, y_test = train_test_split(x_pre_data, y_pre_data, test_size=0.5)

    # 调用模型选择函数
    # classifier_select(x_train,x_test,y_train,y_test)

    # 调用分类模型选择函数
    '''
    模型训练
    '''
    knn_train(x_pre_data,y_pre_data)
    svm_train(x_pre_data, y_pre_data)
    xgboost_train(x_pre_data,y_pre_data)
    decisiontree_train(x_pre_data,y_pre_data)
    randomforest_train(x_pre_data, y_pre_data)
    adaboost_train(x_pre_data, y_pre_data)
    GDBT_train(x_pre_data, y_pre_data)