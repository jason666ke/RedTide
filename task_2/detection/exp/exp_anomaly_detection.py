from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, cal_f1_score, inverse_sliding_win, daily_error, sliding_win_reduce, smooth_anomalies
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
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
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        train_energy_org = inverse_sliding_win(train_energy, self.args.seq_len)
        # train_energy_daily = daily_error(train_energy_org, self.args.seq_len)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_energy_org = inverse_sliding_win(test_energy, self.args.seq_len)
        # test_energy_daily = daily_error(test_energy_org, self.args.seq_len)
        
        # combined_energy = np.concatenate([train_energy_org, test_energy_org], axis=0)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # combined_energy = np.concatenate([train_energy_daily, test_energy_daily], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)
        # print("test_labels:    ", test_labels.shape)
        print("test_energy:     ", test_energy.shape)
        print("test_energy_org:     ", test_energy_org.shape)
        # print("test_energy_daily:     ", test_energy_daily.shape)

        # (3) evaluation on the test set
        # pred = (test_energy_org > threshold).astype(int)
        pred = (test_energy > threshold).astype(int)
        # pred = (test_energy_daily > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        gt_org = inverse_sliding_win(gt, self.args.seq_len)
        gt_daily = sliding_win_reduce(gt_org, self.args.seq_len)

        print("pred:   ", pred.shape)
        print("pred_one:     ", np.count_nonzero(pred == 1))
        print("gt:     ", gt.shape)
        print("gt:     ", gt.shape)
        print("gt_org:     ", gt_org.shape)
        print("gt_daily:     ", gt_daily.shape)
        print("gt_one:     ", np.count_nonzero(gt_daily == 1))

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)
        pred = np.array(pred)
        gt = np.array(gt)
        gt_daily = np.array(gt_daily)
        pred_org = inverse_sliding_win(pred, self.args.seq_len)
        gt_org = inverse_sliding_win(gt, self.args.seq_len)
        pred_daily = sliding_win_reduce(pred_org, self.args.seq_len)

        print("pred: ", pred.shape)
        print("pred_one:     ", np.count_nonzero(pred == 1))
        print("pred_org:    ", pred_org.shape)
        print("pred_org_one:     ", np.count_nonzero(pred_org == 1))
        print("pred_daily:    ", pred_daily.shape)
        print("pred_daily_one:     ", np.count_nonzero(pred_daily == 1))
        print(pred_daily)

        pd.DataFrame(pred_daily).to_csv(self.args.output_path, index=False, header=False)
        
        print("gt:   ", gt.shape)
        print("gt_one:     ", np.count_nonzero(gt == 1))
        print("gt_org:    ", gt_org.shape)
        print("gt_org_one:     ", np.count_nonzero(gt_org == 1))
        print("gt_daily:     ", gt_daily.shape)
        print("gt_one:     ", np.count_nonzero(gt_daily == 1))

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        accuracy_daily, precision_daily, recall_daily, f_score_daily, TP, FP, TN, FN = cal_f1_score(pred_org, gt_org)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        print("Tp : {}, Tn : {}, Fp : {}, Fn : {}, Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            TP, TN, FP, FN, accuracy_daily, precision_daily, recall_daily, f_score_daily))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write("Tp : {}, Tn : {}, Fp : {}, Fn : {}, Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(TP, TN, FP, FN, accuracy_daily, precision_daily, recall_daily, f_score_daily))
        f.write('\n')
        f.write('\n')
        f.close()
        return


    def predict(self, setting):
            _, pred_loader = self._get_data(flag='pred')
            
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

            attens_energy = []

            self.model.eval()
            self.anomaly_criterion = nn.MSELoss(reduce=False)

            attens_energy = []
            for i, (batch_x) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            pred_energy = np.array(attens_energy)
            pred_energy_org = inverse_sliding_win(pred_energy, self.args.seq_len)
            pred_energy_daily = daily_error(pred_energy_org, self.args.seq_len)
            
            combined_energy = pred_energy
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
            print("Threshold :", threshold)
            print("pred_energy:     ", pred_energy.shape)
            print("pred_energy_org:     ", pred_energy_org.shape)
            print("pred_energy_daily:     ", pred_energy_daily.shape)

            pred = (pred_energy > threshold).astype(int)

            print("pred:   ", pred.shape)
            print("pred_one:     ", np.count_nonzero(pred == 1))
            pred = smooth_anomalies(pred)

            pred = np.array(pred)
            pred_org = inverse_sliding_win(pred, self.args.seq_len)
            pred_daily = sliding_win_reduce(pred_org, self.args.seq_len)

            print("pred: ", pred.shape)
            print("pred_one:     ", np.count_nonzero(pred == 1))
            print("pred_org:    ", pred_org.shape)
            print("pred_org_one:     ", np.count_nonzero(pred_org == 1))
            print("pred_daily:    ", pred_daily.shape)
            print("pred_daily_one:     ", np.count_nonzero(pred_daily == 1))
            print(pred_daily)


            pd.DataFrame(pred_daily.reshape(-1, 1)).to_csv(self.args.output_path, index=False, header=False)

            return
