from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, cal_f1_score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
from utils.losses import focal_loss

import pdb

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        self.best_threshold = 0.5 # 初始化最佳阈值

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.CrossEntropyLoss()
        criterion = focal_loss(alpha=0.25, gamma=2)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 获取模型输出
                outputs = self.model(batch_x, padding_mask, None, None)
                # 使用sigmoid函数计算二分概率
                # outputs = torch.sigmoid(outputs)
                
                pred = outputs.detach()
                # loss = criterion(pred, label.long().squeeze().cpu())
                # loss = criterion(pred, label.float().squeeze().cpu())
                loss = criterion(pred, label.float())
                total_loss.append(loss.cpu())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0).cpu()  # 先合并为张量
        trues = torch.cat(trues, 0).cpu()
        # preds = torch.cat(preds, 0).cpu().numpy()
        # trues = torch.cat(trues, 0).cpu().numpy()

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # trues = trues.flatten().cpu().numpy()
        
        # 动态调整分类阈值以优化F1值
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.1):
            predictions = (preds > threshold).cpu().numpy()
            f1,tp,fp,tn,fn = cal_f1_score(predictions, trues)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.best_threshold = best_threshold
        print("Best Threshold: ", self.best_threshold)
        # 将sigmoid输出的概率转化为标签（0或1）
        # predictions = (preds > 0.5).cpu().numpy()
        predictions = (preds > self.best_threshold).numpy()
        trues = trues.numpy()

        accuracy = cal_accuracy(predictions, trues)
        f1,tp,fp,tn,fn = cal_f1_score(predictions, trues)

        self.model.train()
        return total_loss, accuracy, f1, tp, fp, tn, fn

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):

                # pos_samples = (label == 1).sum().item()
                # neg_samples = (label == 0).sum().item()
                # print(f"Batch {i}: Positive samples: {pos_samples}, Negative samples: {neg_samples}")

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 获取模型输出
                outputs = self.model(batch_x, padding_mask, None, None)
                # 使用sigmoid 进行二分类的概率计算
                # outputs = torch.sigmoid(outputs)

                # loss = criterion(outputs, label.long().squeeze(-1))
                # loss = criterion(outputs, label.float().squeeze(-1))
                loss = criterion(outputs, label.float())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_f1, val_tp, val_fp, val_tn, val_fn = self.vali(vali_data, vali_loader, criterion)
            # self.best_threshold = best_threshold
            # test_loss, test_accuracy, test_f1 = self.vali(test_data, test_loader, criterion)

            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
            #     .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Vali F1: {5:.3f} Vali TP: {6:.3f} Vali FP: {7:.3f} Vali TN: {8:.3f} Vali FN: {9:.3f} Best Threshold: {10:.3F}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, val_f1, val_tp, val_fp, val_tn, val_fn, self.best_threshold))
            # early_stopping(-val_accuracy, self.model, path)
            early_stopping(-val_f1, self.model, path) # 早停机制改为对F1敏感
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim,  scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 获取模型输出
                outputs = self.model(batch_x, padding_mask, None, None)
                # 使用sigmoid 进行二分类计算
                outputs = torch.sigmoid(outputs)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # print('test shape:', preds.shape, trues.shape)

        # probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        # predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        # trues = trues.flatten().cpu().numpy()

        # 将sigmoid函数将输出的概率转化为类别标签
        # predictions = (preds > 0.5).cpu().numpy()  # (total_samples,) int class index for each sample
        print("Best Threshold: ", self.best_threshold)
        predictions = (preds > self.best_threshold).cpu().numpy()
        trues = trues.cpu().numpy()
        # trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)
        f1, tp, fp, tn, fn = cal_f1_score(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        print('f1 score:{}'.format(f1))
        print('tp:{}'.format(tp))
        print('fp:{}'.format(fp))
        print('tn:{}'.format(tn))
        print('fn:{}'.format(fn))
        f = open("result_classification.txt", 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('f1 score: {}\n'.format(f1))
        f.write('tp: {}\nfp: {}\ntn: {}\nfn: {}\n'.format(tp, fp, tn, fn))
        f.write('\n')
        f.write('\n')
        f.close()
        return
