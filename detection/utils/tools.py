import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def smooth_anomalies(pred, win=3):
    for i in range(len(pred)):
        if pred[i] == 1:
            for j in range(max(0, i - win), min(len(pred), i + win)):
                pred[j] = 1
    return pred

def sliding_win_reduce(arr, win_size=24):
    """
    通过滑动窗口合并 01 数组，每个窗口大小为 win_size。
    """
    num_win = len(arr) // win_size
    reshaped = arr[:num_win * win_size].reshape(num_win, win_size)
    zero_cnts = np.sum(reshaped == 0, axis=1)
    one_cnts = win_size - zero_cnts

    # result = (one_cnts > zero_cnts).astype(int)
    result = (one_cnts == 24).astype(int)

    return result

def cal_f1_score(y_pred, y_true, win_size=24):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    y_pred = sliding_win_reduce(y_pred)
    y_true = sliding_win_reduce(y_true)
    
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return acc, precision, recall, f1_score, TP, FP, TN, FN

def inverse_sliding_win(arr, win_size=24):
    """
    反滑动窗口，将展平的预测或真实值恢复到原始时间序列形状。
    """
    num_win = len(arr) // win_size
    org_len = num_win + win_size - 1

    arr = arr.reshape(num_win, win_size)

    arr_org = np.zeros(org_len)
    cnt = np.zeros(org_len)

    for i in range(num_win):
        arr_org[i:i+win_size] += arr[i]
        cnt[i:i+win_size] += 1

    cnt[cnt == 0] = 1
    arr_org /= cnt

    return arr_org

def daily_error(arr, win_size=24):
    """
    将 24 小时的误差求和，得到每天的总体误差。
    """
    reshaped = arr.reshape(-1, win_size)
    daily_errors = np.sum(reshaped, axis=1)

    return daily_errors