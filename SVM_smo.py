# _*_ coding:utf-8 _*_
# svm二次规划使用smo进行计算

from numpy import *
import matplotlib.pyplot as plt
import random


def load_test(filename):
    data_mat = []
    label_mat = []
    f = open(filename)
    for i in f.readlines():
        line = i.strip().split('\t')
        data_mat.append([float(line[0]), float(line[1])])
        label_mat.append(float(line[2]))
    return data_mat, label_mat


def show_data(data_mat, label_mat):
    data_plus = []
    data_nage = []
    for i in range(len(data_mat)):
        if label_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_nage.append(data_mat[i])
    data_plus_np = array(data_plus)
    data_nage_np = array(data_nage)
    print(data_plus_np)
    print(transpose(data_plus_np)[0])
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1])
    plt.scatter(transpose(data_nage_np)[0], transpose(data_nage_np)[1])
    plt.show()


def select_diff_j(m, i):  # 选取除了i之外的x_j
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def get_L_H_fun(label_i, label_j, c):  # 获取定义域下的可行解
    if label_i != label_j:
        L = max(0, label_j - label_i)
        H = min(c, c + label_j - label_i)
    else :
        L = max(0, label_j + label_i - c)
        H = min(c, label_j + label_i)
    return L, H


def select_alpha(alpha, L, H):
    if alpha < L:
        alpha = L
    if alpha > H:
        alpha = H
    return alpha


def smo(datamatin, classlabelsin, c, toler, maxiter):
    data_matrix = mat(datamatin)
    class_labels_rix = mat(classlabelsin).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxiter:
        alpha_pairs_change = 0
        for i in range(m):
            fx_i = float(multiply(alphas, class_labels_rix).T * (data_matrix * data_matrix[i, :].T)) + b
            e_i = fx_i - float(class_labels_rix[i])
            if (class_labels_rix[i] * e_i < -toler) and (alphas[i] < c) or (class_labels_rix[i] * e_i > toler) and (alphas[i] > 0):  # 是否需要对alpha进行优化
                j = select_diff_j(m, i)
                fx_j = float(multiply(alphas, class_labels_rix).T * (data_matrix * data_matrix[j, :].T)) + b
                e_j = fx_j - float(class_labels_rix[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # L, H = get_L_H_fun(class_labels_rix[i], class_labels_rix[j], c)
                if class_labels_rix[i] != class_labels_rix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - c)
                    H = min(c, alphas[i] + alphas[j])
                if L == H:
                    print('L == h')
                    continue
                eta = 2 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :]\
                    * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                alphas[j] -= class_labels_rix[j] * (e_i - e_j) / eta
                # print(alphas[j])
                # time.sleep(0.5)
                alphas[j] = select_alpha(alphas[j], L, H)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('alpha_j not moving ')
                    continue
                alphas[i] += class_labels_rix[i] * class_labels_rix[j] * (alpha_j_old - alphas[j])
                b1 = b - e_i - class_labels_rix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T \
                    - class_labels_rix[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - e_j - class_labels_rix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T \
                    - class_labels_rix[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_change += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alpha_pairs_change))
        if alpha_pairs_change == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas


def show_classifer(data_mat, w, b):
    data_plus = []
    data_nage = []
    for i in range(len(data_mat)):
        if label_mat[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_nage.append(data_mat[i])
    data_plus_np = array(data_plus)
    data_nage_np = array(data_nage)
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], alpha=0.7, s=30, c='r')
    plt.scatter(transpose(data_nage_np)[0], transpose(data_nage_np)[1], alpha=0.7, s=30, c='y')
    x1 = max(data_mat)[0]
    x2 = min(data_mat)[0]
    a1, a2 = w
    # print(w, b)
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # print(x1, x2, y1, y2)
    # plt.plot([0, 10], [2, 6])
    plt.show()

    # xp = linspace(x1, x2)
    # yp = -(a1 * xp + b) / a2
    # plt.plot(xp, yp.T, 'b-', linewidth=2.0)
    # plt.show()


def get_w(data_mat, label_mat, alphas):
    x = mat(data_mat)
    label_mat = mat(label_mat).transpose()
    m, n = shape(data_mat)
    w = zeros((n, 1))

    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], x[i, :].T)
    return w


if __name__ == '__main__':
    data_mat, label_mat = load_test('test.txt')
    show_data(data_mat, label_mat)
    b, alphas = smo(data_mat, label_mat, 0.6, 0.001, 40)
    w = get_w(data_mat, label_mat, alphas)
    show_classifer(data_mat, w, b)
