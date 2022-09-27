# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()
        self.__in_features = 0
        self.__out_features = 0
        self.__hidden_units = 0
        self.__W1 = None
        self.__W2 = None
        self.__W3 = None
        self.__h1 = None
        self.__h2 = None
        self.__grad_W1 = None
        self.__grad_W2 = None
        self.__grad_W3 = None
        self.__train_loss = []
        self.__valid_loss = []
        self.__train_acc = []
        self.__valid_acc = []

    def __init_parameter(self, X_train, y_train):
        self.__in_features = X_train.shape[-1]
        self.__out_features = np.max(y_train) + 1
        self.__hidden_units = 2 * self.__in_features + 1
        # 根据数据形状初始化参数形状
        self.__W1 = nn.Parameter(torch.FloatTensor(self.__hidden_units, self.__in_features))
        self.__W2 = nn.Parameter(torch.FloatTensor(self.__hidden_units, self.__hidden_units))
        self.__W3 = nn.Parameter(torch.FloatTensor(self.__out_features, self.__hidden_units))
        # 初始化参数
        nn.init.xavier_normal_(self.__W1)
        nn.init.xavier_normal_(self.__W2)
        nn.init.xavier_normal_(self.__W3)

    def forward(self, x):
        self.__h1 = torch.sigmoid(self.__W1.mm(x))  # W1: h*k, x: k*n, h1: h*k
        self.__h2 = torch.sigmoid(self.__W2.mm(self.__h1))  # W2: h*h, h1: h*k, h2: h*k
        y_score = self.__W3.mm(self.__h2)  # W3: l*h, h2: h*k, y: l*k
        return y_score

    def predict(self, X_test):
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            y_score = self.forward(X_test_tensor.t()).t()
            _, y_pred = torch.max(y_score, 1)
            pred_list = y_pred.numpy().tolist()
            return pred_list

    def __evaluate_valid(self, X_valid, y_valid):
        X_valid_tensor = torch.FloatTensor(X_valid)
        y_valid_tensor = torch.LongTensor(y_valid)
        with torch.no_grad():
            y_valid_score = self.forward(X_valid_tensor.t())
            cross_entropy_valid = nn.CrossEntropyLoss()
            loss_valid = cross_entropy_valid(y_valid_score.t(), y_valid_tensor)
            self.__valid_loss.append(loss_valid)
            return y_valid_score

    def __calculate_acc(self, y_train_true, y_train_score, y_valid_true, y_valid_score):
        _, pred_label = torch.max(y_train_score.t(), 1)
        pred_label = pred_label.numpy()
        self.__train_acc.append(accuracy_score(y_train_true, pred_label))
        _, pred_valid_label = torch.max(y_valid_score.t(), 1)
        pred_valid_label = pred_valid_label.numpy()
        self.__valid_acc.append(accuracy_score(y_valid_true, pred_valid_label))

    def __check_grad(self, print_grad=False):
        def check(name, auto_grad, manual_grad):
            gard_err = manual_grad - auto_grad
            if print_grad:
                print('{}自动推导:'.format(name))
                print(auto_grad)
                print('{}手动推导:'.format(name))
                print(manual_grad)
                print('{}手动推导误差:'.format(name))
                print(gard_err)
            print('{}最大误差：{}'.format(name, torch.max(gard_err)))
            print('{}均方误差: {}'.format(name, gard_err.norm(2).pow(2)))
        check('W1', self.__W1.grad, self.__grad_W1)
        check('W2', self.__W2.grad, self.__grad_W2)
        check('W3', self.__W3.grad, self.__grad_W3)

    def __plot_loss(self):
        plt.plot(range(len(self.__train_loss)), self.__train_loss, label='train loss')
        plt.plot(range(len(self.__valid_loss)), self.__valid_loss, label='valid loss')
        plt.legend()
        plt.xlabel('iter times')
        plt.ylabel('loss')
        plt.savefig('loss.png')
        plt.show()

    def fit(self, X_train, y_train, X_valid, y_valid, batch_size, epoch, learning_rate, check_grad=False):
        self.__init_parameter(X_train, y_train)
        self.__train_loss.clear()
        self.__valid_loss.clear()
        self.__train_acc.clear()
        self.__valid_acc.clear()
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iter = 0
        for i in range(epoch):
            new_epoch = True
            for index, data in enumerate(data_loader):
                print('iter{}'.format(iter))
                iter += 1
                # 计算当前batch的Loss
                y_score = self.forward(data[0].t())

                # torch中的CrossEntropyLoss自带Softmax, 使用计算Softmax之前的值进行计算
                cross_entropy = nn.CrossEntropyLoss()
                loss = cross_entropy(y_score.t(), data[1])
                self.__train_loss.append(float(loss))

                # 手动推导梯度
                y_prob = torch.softmax(y_score.detach(), dim=0)
                grad_l_s3 = y_prob.clone()  # grad_l_s3: l*k
                for idx, col in enumerate(data[1]):
                    grad_l_s3[int(col), idx] -= 1
                grad_s2 = torch.mul(self.__h2, -(self.__h2 - 1))    # grad_s2: h*k
                grad_s1 = torch.mul(self.__h1, -(self.__h1 - 1))    # grad_s1: h*k
                self.__grad_W3 = grad_l_s3.mm(self.__h2.t()) / len(data[0])  # grad_l_s3: l*k, h2.t(): k*h, grad_W3: l*h
                grad_W3_l_s3_s2 = torch.mul(self.__W3.t().mm(grad_l_s3), grad_s2)  # W3.t(): h*l, grad_W3_l_s3_s2: h*k
                self.__grad_W2 = grad_W3_l_s3_s2.mm(self.__h1.t()) / len(data[0])  # h1.t():k*h, grad_W2:h*h
                # W2.t(): h*h, data[0]: k*n, grad_W1: h*n
                self.__grad_W1 = torch.mul(self.__W2.t().mm(grad_W3_l_s3_s2), grad_s1).mm(data[0]) / len(data[0])

                # 检查梯度是否计算正确
                if check_grad:
                    # 自动求梯度
                    self.zero_grad()
                    loss.backward()
                    # 检查梯度，如需要查看梯度具体的矩阵，可以指定print_grad=True
                    self.__check_grad(print_grad=False)

                # 梯度下降更新参数
                # self.__W3 = nn.Parameter(self.__W3 - learning_rate * self.__grad_W3)
                # self.__W2 = nn.Parameter(self.__W2 - learning_rate * self.__grad_W2)
                # self.__W1 = nn.Parameter(self.__W1 - learning_rate * self.__grad_W1)
                self.__W1.data -= learning_rate * self.__grad_W1
                self.__W2.data -= learning_rate * self.__grad_W2
                self.__W3.data -= learning_rate * self.__grad_W3

                # 对验证集进行预测评估
                y_valid_score = self.__evaluate_valid(X_valid, y_valid)

                # 每个Epoch计算一次训练集和验证集的准确率
                if new_epoch:
                    self.__calculate_acc(data[1], y_score, y_valid, y_valid_score)
                    new_epoch = False

            # 每个Epoch打印一次准确率
            print('Epoch {}: train acc:{}, valid acc:{}'.format(i, self.__train_acc[-1], self.__valid_acc[-1]))
        # 绘制loss曲线
        self.__plot_loss()