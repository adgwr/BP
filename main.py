# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random
import torch
from model import BP
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def set_random_seed(seed_value=1024):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


if __name__ == '__main__':
    random_seed = 1024
    set_random_seed(random_seed)
    # 从文件读取数据
    data = pd.read_table('iris.data', sep=',', header=None, names=[0, 1, 2, 3, 'target'])
    data.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}, inplace=True)

    # 按8:1:1划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'],
                                                        test_size=0.2, random_state=random_seed)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                        test_size=0.5, random_state=random_seed)
    # 对数据进行归一化
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_valid = min_max_scaler.transform(X_valid)
    X_test = min_max_scaler.transform(X_test)

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    # 初始化模型并训练
    bp = BP()
    # device = torch.device('cuda')
    # bp.to(device)

    # 用于检查梯度是否正确
    bp.fit(X_train, y_train, X_valid, y_valid, batch_size=32, epoch=1, learning_rate=0.05, check_grad=True)

    # 训练
    # bp.fit(X_train, y_train, X_valid, y_valid, batch_size=32, epoch=3000, learning_rate=0.05, check_grad=False)

    # 分别在训练集、验证集、测试集上进行预测并计算准确率
    y_pred_train = bp.predict(X_train)
    print('train acc:{}'.format(accuracy_score(y_train, y_pred_train)))
    y_pred_valid = bp.predict(X_valid)
    print('valid acc:{}'.format(accuracy_score(y_valid, y_pred_valid)))
    y_pred_test = bp.predict(X_test)
    print('test acc:{}'.format(accuracy_score(y_test, y_pred_test)))







