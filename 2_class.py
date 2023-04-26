import pickle
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score

import numpy as np
import time
import gc
import pandas as pd
import random
from joblib import Parallel, delayed
import warnings, sklearn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=sklearn.exceptions.DataConversionWarning)
n_cores = 10



def split_data(data, test_size=0.25, random_state=0, y_column='insider',
               shuffle=True,
               x_rm_cols=('user', 'day', 'week', 'starttime', 'endtime', 'sessionid',
                          'timeind', 'Unnamed: 0', 'insider'),
               dname='r5.2', normalization='StandardScaler',
               rm_empty_cols=True, by_user=False, by_user_time=False,
               by_user_time_trainper=0.5, limit_ntrain_user=0):
    """
    split data to train and test, can get data by user, seq or random, with normalization builtin
    """
    np.random.seed(random_state)
    random.seed(random_state)

    x_cols = [i for i in data.columns if i not in x_rm_cols]
    if rm_empty_cols:
        x_cols = [i for i in x_cols if len(set(data[i])) > 1]

    infocols = list(set(data.columns) - set(x_cols))

    # output a dict
    out = {}

    # normalization
    if normalization == 'StandardScaler':
        sc = StandardScaler()
    elif normalization == 'MinMaxScaler':
        sc = MinMaxScaler()
    elif normalization == 'MaxAbsScaler':
        sc = MaxAbsScaler()
    else:
        sc = None
    out['sc'] = sc

    # split data randomly by instance
    if not by_user and not by_user_time:
        x = data[x_cols].values
        y_org = data[y_column].values

        y = y_org.copy()
        if 'r6' in dname:
            y[y != 0] = 1

        x_train, x_test, y_train, y_test = train_test_split(x, y_org, test_size=test_size, shuffle=shuffle)

        if 'sc' in out and out['sc'] is not None:
            x_train = sc.fit_transform(x_train)
            out['sc'] = sc
            if test_size > 0:
                x_test = sc.transform(x_test)

    # split data by user
    elif by_user:
        test_users, train_users = [], []
        for i in [j for j in list(set(data['insider'])) if j != 0]:
            uli = list(set(data[data['insider'] == i]['user']))
            random.shuffle(uli)
            ind_i = int(np.ceil(test_size * len(uli)))
            test_users += uli[:ind_i]
            train_users += uli[ind_i:]

        normal_users = list(set(data['user']) - set(data[data['insider'] != 0]['user']))
        random.shuffle(normal_users)
        if limit_ntrain_user > 0:
            normal_ind = limit_ntrain_user - len(train_users)
        else:
            normal_ind = int(np.ceil((1 - test_size) * len(normal_users)))

        train_users += normal_users[: normal_ind]
        test_users += normal_users[normal_ind:]
        x_train = data[data['user'].isin(train_users)][x_cols].values
        x_test = data[data['user'].isin(test_users)][x_cols].values
        y_train = data[data['user'].isin(train_users)][y_column].values
        y_test = data[data['user'].isin(test_users)][y_column].values

        out['train_info'] = data[data['user'].isin(train_users)][infocols]
        out['test_info'] = data[data['user'].isin(test_users)][infocols]

        out['train_users'] = train_users
        if test_size > 0 or limit_ntrain_user > 0:
            out['test_users'] = test_users

        if 'sc' in out and out['sc'] is not None:
            x_train = sc.fit_transform(x_train)
            out['sc'] = sc
            if test_size > 0 or (limit_ntrain_user > 0 and limit_ntrain_user < len(set(data['user']))):
                x_test = sc.transform(x_test)

    # split by user and time
    elif by_user_time:
        train_week_max = by_user_time_trainper * max(data['week'])
        train_insiders = set(data[(data['week'] <= train_week_max) & (data['insider'] != 0)]['user'])
        users_set_later_weeks = set(data[data['week'] > train_week_max]['user'])

        first_part = data[data['week'] <= train_week_max]
        second_part = data[data['week'] > train_week_max]

        first_part_split = split_data(first_part, random_state=random_state, test_size=0,
                                      dname=dname, normalization=normalization,
                                      by_user=True, by_user_time=False,
                                      limit_ntrain_user=limit_ntrain_user,
                                      )

        x_train = first_part_split['x_train']
        y_train = first_part_split['y_train']
        x_cols = first_part_split['x_cols']

        out['train_info'] = first_part_split['train_info']
        out['other_trainweeks_users_info'] = first_part_split['test_info']

        if 'sc' in first_part_split and first_part_split['sc'] is not None:
            out['sc'] = first_part_split['sc']

        out['x_other_trainweeks_users'] = first_part_split['x_test']
        out['y_other_trainweeks_users'] = first_part_split['y_test']
        out['y_bin_other_trainweeks_users'] = first_part_split['y_test_bin']
        out['other_trainweeks_users'] = first_part_split['test_users']  # users in first half but not in train

        real_train_users = set(first_part_split['train_users'])
        real_train_insiders = train_insiders.intersection(real_train_users)
        test_users = list(users_set_later_weeks - real_train_insiders)
        x_test = second_part[second_part['user'].isin(test_users)][x_cols].values
        y_test = second_part[second_part['user'].isin(test_users)][y_column].values
        out['test_info'] = second_part[second_part['user'].isin(test_users)][infocols]
        if ('sc' in out) and (out['sc'] is not None) and (by_user_time_trainper < 1):
            x_test = out['sc'].transform(x_test)

        out['train_users'] = first_part_split['train_users']
        out['test_users'] = test_users

    # get binary data
    y_train_bin = y_train.copy()
    y_train_bin[y_train_bin != 0] = 1

    out['x_train'] = x_train
    out['y_train'] = y_train
    out['y_train_bin'] = y_train_bin
    out['x_cols'] = x_cols
    out['info_cols'] = infocols

    out['test_size'] = test_size

    if test_size > 0 or (by_user_time and by_user_time_trainper < 1) or limit_ntrain_user > 0:
        y_test_bin = y_test.copy()
        y_test_bin[y_test_bin != 0] = 1
        out['x_test'] = x_test
        out['y_test'] = y_test
        out['y_test_bin'] = y_test_bin

    return out




if __name__ == '__main__':

    # 定义标签值
    labels = [0, 1, 2, 3, 4]
    # 将标签0视为正类，将标签1-4视为负类
    positive_class = 4
    negative_classes = [0, 1, 2, 3]
    # 定义需要移除的特征列
    removed_cols = ['user', 'day', 'week', 'starttime', 'endtime', 'sessionid', 'insider']
    # 从csv文件中读取数据
    filename = 'session-r5.2.csv.gz'
    data = pd.read_csv(filename)
    print('data load successfully')

    # 定义模型路径和数据类型
    model_path = 'model.pkl'
    dtype = 'session'

    # 定义四个不同的分类器
    #byday Best parameters found:  {'learning_rate': 0.1, 'min_child_samples': 29, 'n_estimators': 282, 'num_leaves': 22, 'reg_alpha': 0.1, 'reg_lambda': 0.5}
    clfs = {'LightGBM': LGBMClassifier(random_state=42,n_jobs=n_cores,num_leaves=21,min_child_samples= 35,n_estimators=282,learning_rate=0.05,reg_alpha= 0.1, reg_lambda=0.5),
        'LR': LogisticRegression(solver='lbfgs', n_jobs=n_cores),
            'MLP': MLPClassifier(solver='adam'),
            'RF': RandomForestClassifier(n_jobs=n_cores),
            'XGB': XGBClassifier(n_jobs=n_cores),
            }

    # 读取已训练好的模型参数，并将参数赋值给对应的分类器
    with open('params.pkl', 'rb') as f:
        loaded_params = pickle.load(f)
        for c in clfs:
            if c != 'LR' and c != 'LightGBM':
                clfs[c].set_params(**loaded_params[c][dtype])

    # 将数据集拆分为训练集和测试集，并进行标准化等预处理操作
    res = split_data(data, test_size=0.5, random_state=0, y_column='insider',
                     shuffle=True,
                     x_rm_cols=('user', 'day', 'week', 'starttime', 'endtime', 'sessionid',
                                'timeind', 'Unnamed: 0', 'insider'),
                     dname='r5.2', normalization='StandardScaler',
                     rm_empty_cols=True, by_user=False, by_user_time=False,
                     by_user_time_trainper=0.5, limit_ntrain_user=0)

    # print(res)
    # 从预处理后的数据集中获取训练集和测试集数据
    x_train = res['x_train']
    print(len(x_train))
    y_train = res['y_train']
    x_test = res['x_test']
    print(len(x_test))
    y_test = res['y_test']
    y_train_binary = [1 if label == positive_class else 0 for label in y_train]
    y_test_binary = [1 if label == positive_class else 0 for label in y_test]

    # 遍历每个分类器，训练模型并计算性能指标
    for clf_name, clf in clfs.items():
        print('Training', clf_name, 'for positive class', positive_class)
        a = clf.fit(x_train, y_train_binary)
        y_pred_binary = clf.predict(x_test)
        y_prob_binary = clf.predict_proba(x_test)[:, 1]

        # 输出模型得分
        print(a.score(x_test, y_test_binary))

        # 保存模型
        if model_path is not None:
            joblib.dump(clf, clf_name + '-' + model_path)

        # 输出分类报告
        print('Classification Reportfor positive class', positive_class)
        print(classification_report(y_test_binary, y_pred_binary))

        # 计算混淆矩阵及其性能指标
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        print('Confusion Matrix')
        print(cm)

        # 获取TN，FP，FN和TP的值
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn)
        pr = tp / (tp + fp)
        f1 = f1_score(y_test_binary, y_pred_binary)

        print('False Positive Rate:', fpr)
        print('Precision:', pr)
        print('F1 Score:', f1)

        # 计算ROC曲线和AUC值
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob_binary)
        auc_score = auc(fpr, tpr)
        print('ROC AUC Score for positive class', positive_class, ':', auc_score)
        # 绘制ROC曲线图像
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of '+clf_name)
        plt.legend(loc="lower right")
        plt.savefig(clf_name + '-' +'roc_curve.png', dpi=300, bbox_inches='tight')
