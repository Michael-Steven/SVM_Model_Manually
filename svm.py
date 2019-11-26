from numpy import *
from scipy import io
from tqdm import tqdm
import os
import random
import pandas as pd

SVM_TRAINING_SET = 'svm_training_set.csv'
ITER_NUM = 100
C = 2
SIGMA = 0.5
EPS = 0.01


def split_data(input_set):
    _train = []
    _test = []
    for i in range(0, len(input_set)):
        if i % 10 == 1 or i % 10 == 2:
            _test.append(input_set[i])
        else:
            _train.append(input_set[i])
    return array(_train), array(_test)


class SvmModel:
    def __init__(self, data_mat, label, c, eps, sigma):
        self.X = data_mat
        self.label = label
        self.c = c
        self.eps = eps
        self.sigma = sigma
        self.row_num = shape(data_mat)[0]
        self.alpha = zeros(self.row_num)
        self.b = 0
        self.K = zeros((self.row_num, self.row_num))
        print("Calculating train data kernel : ")
        for i in tqdm(range(self.row_num)):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], sigma)


def kernel_trans(x, a, sigma):  # 使用高斯核; x:数据; a：某一行特征数据; sigma:高斯核的参数
    sub = x - a
    mul = multiply(sub, sub)
    mul = sum(mul, axis=1)
    K = exp(-sigma * mul)
    return K


def calculate_p(model, i):
    predict_i = dot(model.K[:, i].T, multiply(model.alpha, model.label)) + model.b
    return predict_i


def check_kkt(model, i):
    fxi = calculate_p(model, i)
    yi = model.label[i]
    if model.alpha[i] < 0 or model.alpha[i] > model.c:
        return False
    elif model.alpha[i] == 0 and fxi * yi >= 1:
        return True
    elif model.alpha[i] == model.c and fxi * yi <= 1:
        return True
    elif fxi * yi == 1:
        return True
    return False


def update_alpha(model, i):
    j = i
    i = random.randrange(i + 1, model.row_num)
    Ei = calculate_p(model, i) - model.label[i]
    Ej = calculate_p(model, j) - model.label[j]
    # 计算eta
    eta = model.K[i, i] + model.K[j, j] - 2 * model.K[i, j]
    # 检查KKT条件，不满足则更新alpha_i和alpha_j
    if eta == 0 or check_kkt(model, i):
        return 0
    alpha_i_old = model.alpha[i].copy()
    alpha_j_old = model.alpha[j].copy()
    # 计算下界L, 上界H
    if model.label[i] != model.label[j]:
        L = max(0, alpha_j_old - alpha_i_old)
        H = min(model.c, model.c + alpha_j_old - alpha_i_old)
    else:
        L = max(0, alpha_j_old + alpha_i_old - model.c)
        H = min(model.c, alpha_j_old + alpha_i_old)
    # 计算新的alpha_j
    alpha_j_new = alpha_j_old + model.label[j] * (Ei - Ej) / eta
    # 判断新的alpha_j的范围并更新alpha_j
    if alpha_j_new > H:
        alpha_j_new = H
    elif alpha_j_new < L:
        alpha_j_new = L
    # 计算新的alpha_i和alpha_j
    model.alpha[i] = alpha_i_old + model.label[i] * model.label[j] * (alpha_j_old - alpha_j_new)
    model.alpha[j] = alpha_j_new
    # 计算b1和b2
    bi = model.label[i] - (calculate_p(model, i) - model.b)
    bj = model.label[j] - (calculate_p(model, j) - model.b)
    # 根据alpha_i和alpha_j更新b
    if 0 <= model.alpha[i] <= model.c:
        model.b = bi
    elif 0 <= model.alpha[j] <= model.c:
        model.b = bj
    else:
        model.b = (bi + bj) / 2.0
    return 1


def smo(data_mat, label, c, eps, iter_num, sigma):
    svm = SvmModel(data_mat, label, c, eps, sigma)
    print("svm model inited")
    cnt = 0
    while cnt < iter_num:
        changed = 0
        alpha_old = svm.alpha.copy()
        for i in tqdm(range(svm.row_num - 1)):
            changed += update_alpha(svm, i)
        sv = nonzero(svm.alpha)[0]
        sv_data = svm.X[sv]
        print("Support Vectors number : %d" % shape(sv_data)[0])
        print("iter: %d, pairs changed %d" % (cnt, changed))
        cnt = cnt + 1
        sub_alpha = (alpha_old - svm.alpha)
        loss = dot(sub_alpha.T, sub_alpha)
        print("loss : %f" % loss)
        if loss < svm.eps:
            break
    return svm.b, svm.alpha


def main():
    if not os.path.exists(SVM_TRAINING_SET):
        print("input file not ready")
    Ratio = ['x1', 'x3', 'x10', 'x11', 'x12']
    Nominal = ['x2', 'x5', 'x6', 'x7', 'x8', 'x9']
    Ordinal = ['x4']
    csv_data = pd.read_csv(SVM_TRAINING_SET)
    y = csv_data['label']
    x = csv_data[Ordinal + Nominal + Ratio]
    train_norm = (x - x.min()) / (x.max() - x.min())
    train_data_mat, test_data_mat = split_data(array(train_norm))
    train_label_mat, test_label_mat = split_data(array(y))

    b, alpha = smo(train_data_mat, train_label_mat, c=C, eps=EPS, iter_num=ITER_NUM, sigma=SIGMA)

    sv = nonzero(alpha)[0]
    sv_data = train_data_mat[sv]
    sv_label = train_label_mat[sv]
    io.savemat('train_result.mat', {'sv': sv, 'sv_data': sv_data, 'sv_label': sv_label, 'alpha': alpha, 'b': b})

    # sv = io.loadmat('train_result.mat')['sv'][0]
    # sv_data = io.loadmat('train_result.mat')['sv_data']
    # sv_label = io.loadmat('train_result.mat')['sv_label'][0]
    # alpha = io.loadmat('train_result.mat')['alpha'][0]
    # b = io.loadmat('train_result.mat')['b'][0][0]

    print("Support Vectors number : %d" % shape(sv_data)[0])

    m, n = shape(test_data_mat)
    tp = fp = tn = fn = 0
    for i in range(m):
        kernel_v = kernel_trans(sv_data, test_data_mat[i, :], sigma=SIGMA)
        predict = dot(kernel_v.T, multiply(alpha[sv], sv_label)) + b
        if sign(predict) == 1 and sign(test_label_mat[i]) == 1:
            tp = tp + 1
        elif sign(predict) == 1 and sign(test_label_mat[i]) == -1:
            fp = fp + 1
        elif sign(predict) == -1 and sign(test_label_mat[i]) == -1:
            tn = tn + 1
        elif sign(predict) == -1 and sign(test_label_mat[i]) == 1:
            fn = fn + 1
    print("TP = %d" % tp)
    print("FP = %d" % fp)
    print("TN = %d" % tn)
    print("FN = %d" % fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    print("F1 = %f" % (2.0 * precision * recall / (precision + recall)))


if __name__ == '__main__':
    main()
