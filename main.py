import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from torchvision import transforms
import tensorflow as tf

from torch.utils.mobile_optimizer import optimize_for_mobile
torch.set_grad_enabled(False)


def rmse_function_torch(x):
    return torch.sqrt(torch.mean(x ** 2))


# Данные
train_dataset = datasets.MNIST(root='data', train=True,
                               transform=transforms.ToTensor(), download=True)

Y = train_dataset.targets
X = train_dataset.data.flatten(1)
T = torch.eye(10)[Y]
X = X.to(torch.float64)
T = T.to(torch.float64)



class SCNTrain(torch.nn.Module):
    def __init__(self, lambdas, max_neurons, reconfig_number, Tt):
        super(SCNTrain, self).__init__()
        self.lambdas = lambdas
        self.max_neurons = max_neurons
        self.reconfig_number = reconfig_number
        self.Tt = Tt

    def forward(self, X):
        Y = self.Tt
        d = X.shape[1]
        e = Y

        H = torch.empty((X.shape[0], self.max_neurons), dtype=torch.float64)
        W = torch.empty((d, 0), dtype=torch.float64)    # веса полученной сети
        b = torch.empty(0, dtype=torch.float64)    # смещения полученной сети
        beta = torch.empty(0, dtype=torch.float64)

        for k in range(self.max_neurons):

            # Генерируем случайным образом набор весов
            W_random = []
            b_random = []
            for L in self.lambdas:
                WL = L * (2 * torch.rand(d, self.reconfig_number, dtype=torch.float64) - 1)
                bL = L * (2 * torch.rand(1, self.reconfig_number, dtype=torch.float64) - 1)
                W_random.append(WL)
                b_random.append(bL)
            W_random = torch.hstack(W_random)
            b_random = torch.hstack(b_random)

            # Находим активацию
            h = torch.special.expit(X @ W_random + b_random)

            # находим лучшие веса
            v_values = (e.T @ h) ** 2 / torch.sum(h * h, dim=0)
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)
            h_c = h[:, best_idx]

            H[:, k] = h_c

            # сохраняем веса
            W_c = W_random[:, best_idx]
            W_c = torch.unsqueeze(W_c, dim=-1)
            b_c = b_random[:, best_idx]
            W = torch.concatenate((W, W_c), dim=-1)
            b = torch.concatenate((b, b_c), dim=0)

            # Находим выходной вес
            beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ Y

            # Находим выход сети
            y = H[:, 0: (k + 1)] @ beta

            # Рассчитываем вектор ошибки
            e = Y - y
            rmse = torch.sqrt(torch.mean(e ** 2))
            print(k, rmse)

        return (W, b, beta)


class SCNUTrain(torch.nn.Module):
    def __init__(self, lambdas, max_neurons, reconfig_number):
        super(SCNUTrain, self).__init__()
        self.lambdas = lambdas
        self.max_neurons = max_neurons
        self.reconfig_number = reconfig_number

    def forward(self, X, Y):
        d = X.shape[1]
        e = Y

        H = torch.empty((X.shape[0], self.max_neurons), dtype=torch.float64)
        W = torch.empty((d, 0), dtype=torch.float64)    # веса полученной сети
        b = torch.empty(0, dtype=torch.float64)    # смещения полученной сети
        beta = torch.empty(0, dtype=torch.float64)

        Q = torch.zeros((d, 0))
        R = torch.zeros((2, 2))
        QT = torch.zeros((3, 10))

        for k in range(self.max_neurons):

            # Генерируем случайным образом набор весов
            W_random = []
            b_random = []
            for L in self.lambdas:
                WL = L * (2 * torch.rand(d, self.reconfig_number, dtype=torch.float64) - 1)
                bL = L * (2 * torch.rand(1, self.reconfig_number, dtype=torch.float64) - 1)
                W_random.append(WL)
                b_random.append(bL)
            W_random = torch.hstack(W_random)
            b_random = torch.hstack(b_random)

            # Находим активацию
            h = torch.special.expit(X @ W_random + b_random)

            # находим лучшие веса
            v_values = (e.T @ h) ** 2 / torch.sum(h * h, dim=0)
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)
            h_c = h[:, best_idx]

            H[:, k] = h_c

            # сохраняем веса
            W_c = W_random[:, best_idx]
            W_c = torch.unsqueeze(W_c, dim=-1)
            b_c = b_random[:, best_idx]
            W = torch.concatenate((W, W_c), dim=-1)
            b = torch.concatenate((b, b_c), dim=0)

            # Находим выходной вес
            Ns = 2
            if k < Ns:
                beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ Y
            elif k == Ns:
                Q, R = torch.linalg.qr(H[:, 0: (k + 1)])
                QT = Q.T @ Y
                beta = torch.linalg.solve_triangular(R, QT, upper=True)
                print(QT.shape)
            else:
                r1 = (Q.T @ h_c).squeeze()
                t = h_c.squeeze() - Q @ r1
                r2 = torch.linalg.norm(t)
                q = t / r2

                Q = torch.hstack([Q, q.unsqueeze(1)])
                r = torch.hstack([R, r1.unsqueeze(1)])
                r3 = torch.cat([torch.zeros((k, 1)), r2.unsqueeze(0).unsqueeze(0)])
                R = torch.vstack([r, r3.T])

                QT = torch.vstack([QT, (Q[:, -1] @ Y).unsqueeze(1).T])
                beta = torch.linalg.solve_triangular(R, QT, upper=True)

            # Находим выход сети
            y = H[:, 0: (k + 1)] @ beta

            # Рассчитываем вектор ошибки
            e = Y - y
            rmse = torch.sqrt(torch.mean(e ** 2))
            print(k, rmse)

        return W, b, beta


class BSCNUTrain(torch.nn.Module):
    def __init__(self, lambdas, max_neurons, reconfig_number, init_batch_size, batch_increment):
        super(BSCNUTrain, self).__init__()
        self.lambdas = lambdas
        self.max_neurons = max_neurons
        self.reconfig_number = reconfig_number
        self.init_batch_size = init_batch_size
        self.batch_increment = batch_increment

    def forward(self, X, Y):
        d = X.shape[1]
        e = Y

        H = torch.empty((X.shape[0], self.max_neurons), dtype=torch.float64)
        W = torch.empty((d, 0), dtype=torch.float64)    # веса полученной сети
        b = torch.empty(0, dtype=torch.float64)    # смещения полученной сети
        beta = torch.empty(0, dtype=torch.float64)

        Q = torch.zeros((d, 0))
        R = torch.zeros((2, 2))
        QT = torch.zeros((3, 10))

        y = e

        batch_size = self.init_batch_size
        for k in range(self.max_neurons):

            indices = torch.randperm(X.shape[0])[:batch_size]
            Xk = X[indices, :]
            Tk = Y[indices, :]

            if k == 0:
                ek = Tk.squeeze()
            else:
                ek = Tk.squeeze() - y[indices, :].squeeze()

            # Генерируем случайным образом набор весов
            W_random = []
            b_random = []
            for L in self.lambdas:
                WL = L * (2 * torch.rand(d, self.reconfig_number, dtype=torch.float64) - 1)
                bL = L * (2 * torch.rand(1, self.reconfig_number, dtype=torch.float64) - 1)
                W_random.append(WL)
                b_random.append(bL)
            W_random = torch.hstack(W_random)
            b_random = torch.hstack(b_random)

            # Находим активацию
            h = torch.special.expit(Xk @ W_random + b_random)

            # находим лучшие веса
            v_values = (ek.T @ h) ** 2 / torch.sum(h * h, dim=0)
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)

            W_c = W_random[:, best_idx]
            W_c = torch.unsqueeze(W_c, dim=-1)
            b_c = b_random[:, best_idx]
            h_c = torch.special.expit(X @ W_c + b_c).squeeze()
            H[:, k] = h_c

            # сохраняем веса
            W = torch.concatenate((W, W_c), dim=-1)
            b = torch.concatenate((b, b_c), dim=0)

            # Находим выходной вес
            Ns = 2
            if k < Ns:
                beta = torch.linalg.inv(H[:, 0: (k + 1)].T @ H[:, 0: (k + 1)]) @ H[:, 0: (k + 1)].T @ Y
            elif k == Ns:
                Q, R = torch.linalg.qr(H[:, 0: (k + 1)])
                QT = Q.T @ Y
                beta = torch.linalg.solve_triangular(R, QT, upper=True)
                print(QT.shape)
            else:
                r1 = (Q.T @ h_c).squeeze()
                t = h_c.squeeze() - Q @ r1
                r2 = torch.linalg.norm(t)
                q = t / r2

                Q = torch.hstack([Q, q.unsqueeze(1)])
                r = torch.hstack([R, r1.unsqueeze(1)])
                r3 = torch.cat([torch.zeros((k, 1)), r2.unsqueeze(0).unsqueeze(0)])
                R = torch.vstack([r, r3.T])

                QT = torch.vstack([QT, (Q[:, -1] @ Y).unsqueeze(1).T])
                beta = torch.linalg.solve_triangular(R, QT, upper=True)

            # Находим выход сети
            y = H[:, 0: (k + 1)] @ beta

            # Рассчитываем вектор ошибки
            e = Y - y
            rmse = torch.sqrt(torch.mean(e ** 2))
            batch_size += self.batch_increment
            print(k, rmse)

        return W, b, beta


class OSCNTrain(torch.nn.Module):
    def __init__(self, lambdas, max_neurons, reconfig_number, Tt):
        super(OSCNTrain, self).__init__()
        self.lambdas = lambdas
        self.max_neurons = max_neurons
        self.reconfig_number = reconfig_number
        self.Tt = Tt

    def orthogonalize(A, b):
        r1 = (A.T @ b)
        t = b - A @ r1
        r2 = torch.norm(t, dim=0)
        return t / r2

    def forward(self, X):
        Y = self.Tt
        d = X.shape[1]
        e = self.Tt

        H = torch.empty((X.shape[0], self.max_neurons), dtype=torch.float64)
        W = torch.empty((d, 0), dtype=torch.float64)    # веса полученной сети
        b = torch.empty(0, dtype=torch.float64)    # смещения полученной сети
        beta = torch.empty(0, dtype=torch.float64)

        for k in range(self.max_neurons):

            # Генерируем случайным образом набор весов
            W_random = []
            b_random = []
            for L in self.lambdas:
                WL = L * (2 * torch.rand(d, self.reconfig_number, dtype=torch.float64) - 1)
                bL = L * (2 * torch.rand(1, self.reconfig_number, dtype=torch.float64) - 1)
                W_random.append(WL)
                b_random.append(bL)
            W_random = torch.hstack(W_random)
            b_random = torch.hstack(b_random)

            # Находим активацию
            h = torch.special.expit(X @ W_random + b_random)

            if k == 0:
                v = h
            else:
                r1 = (H[:, :k].T @ h)
                t = h - H[:, :k] @ r1
                r2 = torch.norm(t, dim=0)
                v = t / r2

            v_values = (e.T @ h) ** 2
            # v_values = torch.nan_to_num(v_values, nan=-1)
            v_values_tmp = torch.mean(v_values, dim=0)
            best_idx = torch.argmax(v_values_tmp)


            h_c = v[:, best_idx]

            if k == 0:
                h_c = h_c / torch.norm(h_c)
                beta = e.T @ h_c
                beta = beta.unsqueeze(0)
            else:
                beta_c = e.T @ h_c
                beta = torch.concatenate((beta, beta_c.unsqueeze(0)), dim=0)

            H[:, k] = h_c

            W_c = W_random[:, best_idx].unsqueeze(dim=-1)
            b_c = b_random[:, best_idx]
            W = torch.concatenate((W, W_c), dim=-1)
            b = torch.concatenate((b, b_c), dim=0)

            e = e - h_c.unsqueeze(1) * beta[-1, None, :]
            rmse = torch.sqrt(torch.mean(e ** 2))
            print(k, rmse)
        H = torch.special.expit(X @ W + b)
        beta = torch.linalg.inv(H.T @ H) @ H.T @ Y

        return W, b, beta



# Параметры SCN
Lambdas = [0.001, 0.005, 0.01, 0.05]
max_neurons = 100
reconfig_number = 100
batch_size = 5000
batch_increment = (X.shape[0] - batch_size) // max_neurons

SCN_train = SCNTrain(Lambdas, max_neurons, reconfig_number, T)
SCNU_train = SCNUTrain(Lambdas, max_neurons, reconfig_number)
BSCNU_train = BSCNUTrain(Lambdas, max_neurons, reconfig_number, batch_size, batch_increment)
OSCN_train = OSCNTrain(Lambdas, max_neurons, reconfig_number, T)

import time
start_time = time.time()

# script
# torchscript_model1 = torch.jit.script(SCN_train)
# optimized_torchscript_model1 = optimize_for_mobile(torchscript_model1, backend='CPU')
# W, b, beta = optimized_torchscript_model1(X, T)
# print()

# torchscript_model2 = torch.jit.script(SCNU_train)
# optimized_torchscript_model2 = optimize_for_mobile(torchscript_model2, backend='CPU')
# W, b, beta = optimized_torchscript_model2(X, T)
# print()

# torchscript_model3 = torch.jit.script(BSCNU_train)
# optimized_torchscript_model3 = optimize_for_mobile(torchscript_model3, backend='CPU')
# W, b, beta = optimized_torchscript_model3(X, T)
# print()

torchscript_model4 = torch.jit.script(OSCN_train)
optimized_torchscript_model4 = optimize_for_mobile(torchscript_model4, backend='CPU')
W, b, beta = torchscript_model4(X)
# optimized_torchscript_model4.save("asd.pt")
print("--- %s seconds ---" % (time.time() - start_time))