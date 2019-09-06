# _*_ coding:utf-8 _*_

from numpy import *
import matplotlib.pyplot as plt
import random
import time


class optstruct:
    def __init__(self, dataset, labelset, c, toler, sigma):
        self.x = dataset    # 训练集
        self.label_mat = labelset   # yi
        self.c = c  # 限制α
        self.toler = toler  # 误差容忍
        self.m = shape(self.x)[0]   # 训练样本长度
        self.alphas = mat(zeros((self.m, 1)))   # 存储拉格朗日乘子
        self.b = 0
        self.ecache = mat(zeros((self.m, 2)))   # 存储误差
        self.sigma = sigma  # 高斯核的参数
        self.k = mat(zeros((self.m, self.m)))   # 核函数内积，既k(i, i)
        for i in range(self.m):
            self.k[:, i] = calc_kernel(self.x, self.x[i, :])


def calc_kernel(x, a):  # 计算核内积
    m, n = shape(x)
    k = mat(zeros((m, 1)))
    for i in range(m):
        delta = x[i, :] - a
        k[i] = delta * delta.T
    k = exp(k / (-1 * 10000.0**2))  # 高斯核函数: f(x) = exp(-||x_i - x_j||^2 / 2 * sigma**2)
    return k


def calc_ek(os, k):     # 计算预测偏差的值
    # f(x_i) = w.T * X + b = (Σ α_i * y_i * x_i ).T * X +b
    fx_k = float(multiply(os.alphas, os.label_mat).T * os.k[:, k] + os.b)
    e_k = fx_k - float(os.label_mat[k]) # δ = fx_k - 标签值
    return e_k


def select_random_j(i, os): # 随机选择α2
    j = i
    while j == i:   # 直到α不等于i
        j = int(random.uniform(0, os.m))    # 随机选择
    return j


def select_j(os, i, ei):    # 选择使得优化步长最大的α2
    max_deltae = 0  # 标记最大Δ
    return_k = -1   # 记录最大Δ的下标
    return_ek = 0   # 记录最大Δ的值
    os.ecache[i] = [1, ei]  # 更新i下标的误差
    validecachelist = nonzero(os.ecache[:, 0].A)[0]     # 得到误差不为0的最大ek进行优化
    if len(validecachelist) > 1:    # 如果有误差不为零则进入循环
        for k in validecachelist:
            if k == i:
                continue
            ek = calc_ek(os, k) # 预测值
            deltae = abs(ek - ei)   # 计算误差
            if deltae > max_deltae: # 如果当前误差大于标记的最大误差，则替换之
                max_deltae = deltae
                return_ek = ek
                return_k = k
    else:   # 否则随机选择一个J进行优化
        return_k = select_random_j(i, os)
        return_ek = calc_ek(os, return_k)
    return return_k, return_ek


def update_ek(os, k):  # 更新误差值
    e_k = calc_ek(os, k)
    os.ecache[k] = [1, e_k]


# def get_L_H_fun(label_i, label_j, c):  # 获取定义域下的可行解
#     if label_i != label_j:
#         L = max(0, label_j - label_i)
#         H = min(c, c + label_j - label_i)
#     else :
#         L = max(0, label_j + label_i - c)
#         H = min(c, label_j + label_i)
#     return L, H


def clipalpha(a_j, L, H):  # 获得可行域下的α
    if a_j < L:
        a_j = L
    if a_j > H:
        a_j = H
    return a_j


def inner(i, os):  # 内循环进行alpha的求解
    e_i = calc_ek(os, i)    # 计算α_i的预测值
    # print(e_i)
    # 判断是否需要更新
    if (e_i * os.label_mat[i] < -os.toler) and (os.alphas[i] < os.c) or \
         (e_i * os.label_mat[i] > os.toler) and (os.alphas[i] > 0):
        j, e_j = select_j(os, i, e_i)
        j_old = os.alphas[j].copy()     # 保存原来的alpha
        i_old = os.alphas[i].copy()     # 保存原来的alpha
        # L, H = get_L_H_fun(os.label_mat[i], os.label_mat[j], os.c)
        if os.label_mat[i] != os.label_mat[j]:  # 计算出α的取值范围
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.c)
            H = min(os.c, os.alphas[i] + os.alphas[j])
        eta = 2.0 * os.k[i, j] - os.k[i, i] - os.k[j, j]    # 计算η
        if eta >= 0:
            print('eta>=0')
            return 0
        os.alphas[j] -= os.label_mat[j] * (e_i - e_j) / eta # 获得更新后的α_j
        os.alphas[j] = clipalpha(os.alphas[j], L, H)    # 对α进行剪枝
        if abs(os.alphas[j] - j_old) < 0.00001: # 如果α变化太小，则认为已经更新完毕
            print('alpha_j not moving enough')
            return 0
        # 由α_j得到α_i ， 公式: (old)α1 * y1 + (old)α2 * y2 = (new)α1 * y1 + (new)α2 * y2 →→→\
        # (new)α1 = (old)α1 + [(old)α2 - (new)α2] * y2 * y1
        os.alphas[i] += os.label_mat[i] * os.label_mat[j] * (j_old - os.alphas[j])
        update_ek(os, i)    # 更新误差
        b1 = os.b - e_i - os.label_mat[i] * (os.alphas[i] - i_old) * os.k[i, i] \
             - os.label_mat[j] * (os.alphas[j] - j_old) * os.k[i, j]    # 计算b1
        b2 = os.b - e_j - os.label_mat[i] * (os.alphas[i] - i_old) * os.k[i, j] \
             - os.label_mat[j] * (os.alphas[j] - j_old) * os.k[j, j]    # 计算b2
        if 0 < os.alphas[i] < os.c:
            os.b = b1
        elif 0 < os.alphas[j] < os.c:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo(dataset, labelset, c, to, maxiter):     # 主循环
    os = optstruct(mat(dataset), mat(labelset).transpose(), c=c, toler=to, sigma=1)     # 初始化数据
    iter = 0    # 循环变量
    alphapairchange = 0     # α是否变化
    entire = True   # 控制跳转
    while(iter < maxiter) and (alphapairchange > 0) or entire:  # 当循环次数小于给定的训练次数 或者 α更新过了 或者 进入完整遍历的时候
        alphapairchange = 0
        if entire:  # 对整个数据集进行训练
            for i in range(os.m):
                alphapairchange += inner(i, os)
            iter += 1
        else:   # 对非边界α值遍历 也就是不在边界0，c上的值
            nonbound = nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            for i in nonbound:
                alphapairchange += inner(i, os)
            iter += 1
        if entire:
            entire = False
        elif alphapairchange == 0:
            entire = True
    return os.b, os.alphas


def load_test(filename):    # 加载数据集
    data_mat = []
    label_mat = []
    f = open(filename)
    for i in f.readlines():
        line = i.strip().split('\t')
        data_mat.append([float(line[0]), float(line[1])])
        label_mat.append(float(line[2]))
    return data_mat, label_mat


# def show_classifer(data_mat, label_mat,  w, b):     # 展示
#     data_plus = []
#     data_nage = []
#     for i in range(len(data_mat)):
#         if label_mat[i] > 0:
#             data_plus.append(data_mat[i])
#         else:
#             data_nage.append(data_mat[i])
#     data_plus_np = array(data_plus)
#     data_nage_np = array(data_nage)
#     plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], alpha=0.7, s=30, c='r')
#     plt.scatter(transpose(data_nage_np)[0], transpose(data_nage_np)[1], alpha=0.7, s=30, c='y')
#     x1 = max(data_mat)[0]
#     x2 = min(data_mat)[0]
#     a1, a2 = w
#     # print(w, b)
#     b = float(b)
#     a1 = float(a1[0])
#     a2 = float(a2[0])
#     y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
#     plt.plot([x1, x2], [y1, y2])
#     # print(x1, x2, y1, y2)
#     # plt.plot([0, 10], [2, 6])
#     plt.show()
#
#     # xp = linspace(x1, x2)
#     # yp = -(a1 * xp + b) / a2
#     # plt.plot(xp, yp.T, 'b-', linewidth=2.0)
#     # plt.show()


# def show(data_mat, label_mat):
#     data_plus = []
#     data_nage = []
#     for i in range(len(data_mat)):
#         if label_mat[i] > 0:
#             data_plus.append(data_mat[i])
#         else:
#             data_nage.append(data_mat[i])
#     data_plus_np = array(data_plus)
#     data_nage_np = array(data_nage)
#     plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], alpha=0.7, s=30, c='r')
#     plt.scatter(transpose(data_nage_np)[0], transpose(data_nage_np)[1], alpha=0.7, s=30, c='y')
#     plt.show()


def get_w(data_mat, label_mat, alphas):     # 根据计算出来的α得到w向量 公式 : W = Σ α_i * y_i * x_i
    x = mat(data_mat)
    label_mat = mat(label_mat).transpose()
    m, n = shape(x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


if __name__ == '__main__':
    data_set, label_set = load_test('testSetRBF.txt')
    # show(data_set, label_set)
    b, alphas = smo(data_set, label_set, 200, 0.0001, 100)
    data_mat = mat(data_set)
    label_mat = mat(label_set).transpose()
    svind = nonzero(alphas.A > 0)[0]
    svs = data_mat[svind]
    svlabel = label_mat[svind]
    print('支撑向量个数:%d' % shape(svs)[0])
    m, n = shape(data_mat)
    error = 0
    for i in range(m):
        k = calc_kernel(svs, data_mat[i])
        predict = k.T * multiply(svlabel, alphas[svind]) + b
        if sign(predict) != sign(label_mat[i]):
            error += 1
    print('errorrate:%f', float(error/m))
    # w = get_w(data_set, label_set, alphas)
    # show_classifer(data_set, label_set, w, b)
    test_data_set, test_label_set = load_test('RBF2.txt')
    t_d = mat(test_data_set)
    t_l = mat(test_label_set).transpose()
    # print(mat(test_label_set))
    m, n = shape(t_d)
    error = 0
    for i in range(m):
        k = calc_kernel(svs, t_d[i])
        pre = k.T * multiply(svlabel, alphas[svind]) + b
        # print(sign(t_l[i]))
        if sign(pre) != sign(t_l[i]):
            error += 1
    print('test_errorrate:%f', float(error / m))


'''
|------------------------------------|
|高斯核参数σ|训练集错误率|测试集错误率|
|    1.0    |  0.0 - 0.1 |0.07 - 0.08|
|    1.3    |  0.09      |0.18       |
|    1.4    |  0.01      |0.02       |
|    1.6    |  0.05      |0.05-0.06  |
|    1.7    |  0.07      |0.08       |
|    2.0    |  0.15      |0.16       |
|    3.0    |  0.03      |0.13       |
|    10.0   |  0.52      |0.48       |
|    50.0   |  0.29      |0.51       |
|   100.0   |  0.39      |0.39       |
|  1000.0   |  0.33      |0.41       |
| 10000.0   |  0.30      |0.51       |
|------------------------------------|

'''