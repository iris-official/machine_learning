# -*- coding: UTF-8 -*- 
# @Author: Iris_biubiu
# @Time  : 2018-11
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def data_explore(data):
    explore_data = data
    return explore_data
def data_process(data):
    processed_data = data
    return processed_data
def classifier_select(x_train,x_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import  DecisionTreeClassifier
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
        name = classifier.__class__.__name__
        clf.fit(x_train,y_train)
        # 对测试集预测
        print("☆"*15+name+"☆"*15)
        y_pred = clf.predic(x_test)
        y_pred_proda = clf.predict_proba(x_test)[:,1]

        acc = accuracy_score(y_test,y_pred)
        print("acc" + str(acc))

        auc_value = roc_auc_score(y_test,y_pred_proda)
        print("auc_value:" + str(auc_value))

        result_clf = pd.Series({"classifier": name, 'acc': acc ,'auc_value': auc_value})
        result = result.append(result_clf, ignore_index=True)
        np.savetxt("分类结果_" + name + ".txt", pd.DataFrame(y_pred), fmt='%s', newline='/n')

def xgboost_train(x_pre_data, y_pre_data):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    clf = XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=clf, param_grid=cv_params, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(x_pre_data, x_pre_data)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

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

    x_pre_data = pre_data.drop(features_goal, axis=1)
    y_pre_data = pre_data[features_goal]
    x_train, x_test, y_train, y_test = train_test_split(x_pre_data, y_pre_data, test_size=0.5, random_stats=1)

    # 调用模型选择函数
    classifier_select(x_pre_data, y_pre_data)

    # 调用分类模型选择函数
    '''
    模型训练
    '''


