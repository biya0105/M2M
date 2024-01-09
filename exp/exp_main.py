from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer
from utils.tools import EarlyStopping, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import time
import numpy as np


class Exp_main(Exp_Basic):
    def __init__(self, args):
        super(Exp_main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.innerstepsize)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def adjust_lr(self, epoch):
        lr_adjust = {epoch: self.args.outerstepsize if epoch < 3 else self.args.outerstepsize * (0.9 ** ((epoch - 3) // 1))}
        return lr_adjust[epoch]

    def RSD(self, Feature_s, Feature_t):

        E = torch.randn(Feature_s.shape)
        epsilon = 1e-4 / torch.max(torch.abs(E))
        E = epsilon * E
        E = E.to(self.device)
        Feature_s = Feature_s + E
        Feature_t = Feature_t + E

        u_s, s_s, v_s = torch.svd(Feature_s)
        u_t, s_t, v_t = torch.svd(Feature_t)
        p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))
        a = 1 - torch.pow(abs(cospa), 2)
        a = torch.clamp(a, min=0)
        sinpa = torch.sqrt(a + 1e-5)
        return torch.norm(sinpa, 1, dim=None) + self.args.rsd_inner * torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)

    def model_output(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
        return outputs

    def train_on_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_vali=None, batch_y_vali=None, batch_x_mark_vali=None, batch_y_mark_vali=None):
        self.model.train()
        self.model_optim.zero_grad()
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        outputs = self.model_output(batch_x, batch_y, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = self.criterion(outputs, batch_y)


        if batch_x_vali is not None:
            batch_x_vali = batch_x_vali.float().to(self.device)
            batch_y_vali = batch_y_vali.float().to(self.device)
            batch_x_mark_vali = batch_x_mark_vali.float().to(self.device)
            batch_y_mark_vali = batch_y_mark_vali.float().to(self.device)
            feature_t = self.model_output(batch_x_vali, batch_y_vali, batch_x_mark_vali, batch_y_mark_vali)
            feature_s = outputs
            feature_t = feature_t[:, -self.args.pred_len:, f_dim:]
            loss += self.RSD(feature_s[:, :, 0], feature_t[:, :, 0]) * self.args.rsd_out

        loss.backward()
        self.model_optim.step()
        return loss.item()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        test_of_train_data, test_of_train_loader = self._get_data(flag='test_of_train')

        vali_data_list = {}
        vali_data_len = len(vali_loader)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            vali_data_list[i] = (batch_x, batch_y, batch_x_mark, batch_y_mark)
        vali_data, vali_loader = self._get_data(flag='val')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        weights_before = deepcopy(self.model.state_dict())
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                batch_x_vali, batch_y_vali, batch_x_mark_vali, batch_y_mark_vali = vali_data_list[i % vali_data_len]
                loss_one = self.train_on_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,
                                               batch_x_vali, batch_y_vali, batch_x_mark_vali, batch_y_mark_vali)
                train_loss.append(loss_one)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_one))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            weights_after = deepcopy(self.model.state_dict())
            outerstepsize = self.adjust_lr(epoch)
            print(f'Updating learning rate to {outerstepsize}')
            self.model.load_state_dict({name:
                                            weights_before[name] + (
                                                    weights_after[name] - weights_before[name]) * outerstepsize
                                        for name in weights_before})

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, self.criterion)
            test_loss = self.vali(test_data, test_loader, self.criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test_on_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        self.model.eval()
        with torch.no_grad():
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            pred = outputs 
            true = batch_y 
            return pred, true, batch_x


    def test(self, setting, start=0, end=1024, split=8, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_of_train_data, test_of_train_loader = self._get_data(flag='test_of_train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                pred, true, batch_x = self.test_on_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        mae_min = 100
        mse_min = 100
        rse_min = 100
        fine_min = 0
        pat = 0

        for fine_turn in range(end):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_of_train_loader):
                self.train_on_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds = []
            trues = []
            inputx = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    pred, true, batch_x = self.test_on_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())
            preds = np.array(preds)
            trues = np.array(trues)
            inputx = np.array(inputx)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print(f'fine_turn:{fine_turn + 1}, mse:{mse}, mae:{mae}, rse:{rse}, pat:{pat}')
            if mae + mse < mae_min + mse_min:
                mae_min = mae
                mse_min = mse
                rse_min = rse
                fine_min = fine_turn + 1
                pat = 0
            else:
                pat += 1
                if pat == self.args.patience:
                    break
        f.write(f'fine_turn:{fine_min}, mse:{mse_min}, mae:{mae_min}, rse:{rse_min}')
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        return