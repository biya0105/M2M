import math
import os
import random

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Water_Meta(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='water.npy',
                 target='PH', scale=True, timeenc=0, freq='w'):
        if size == None:
            self.seq_len = 52
            self.label_len = 26
            self.pred_len = 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val', 'test_of_train']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'test_of_train': 3}
        self.set_type = type_map[flag]


        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))

        border_domain1 = [0, 100, 100, 100]
        border_domain2 = [100, 120, 120, 120]
        border_x = [0, 0, 0, 0]
        border_y = [110, 90, 110, 90]

        self.border_domain1 = border_domain1[self.set_type]
        self.border_domain2 = border_domain2[self.set_type]
        self.border_x = border_x[self.set_type]
        self.border_y = border_y[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[:, :, 0:4] 
        elif self.features == 'S':
            target_map = {'PH': 0, 'DO': 1, 'COD': 2, 'NH3-N': 3}
            set_target = target_map[self.target]
            df_data = np.expand_dims(df_raw[:, :, set_target], 2)

        if self.scale:
            data_tp = df_data.transpose(1, 0, 2)
            trans_data = []
            for i in np.arange(120):
                self.scaler.fit(data_tp[i])
                trans_data.append(self.scaler.transform(data_tp[i]).tolist())
            data = np.array(trans_data).transpose(1, 0, 2)
            np.save('./dataset/water_norm.npy', data)
        else:
            data = df_data

        if self.timeenc == 0:
            print(self.timeenc)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            timestamp = time_features(pd.to_datetime(np.arange(0,110), unit='W', origin=pd.Timestamp('2013-01-05')), freq='W')
            timestamp = timestamp.transpose(1, 0)
            data_stamp = []
            for i in np.arange(120):
                data_stamp.append(timestamp.tolist())
            data_stamp = np.array(data_stamp).transpose(1, 0, 2) 


        self.data_x = data[self.border_x:self.border_y, self.border_domain1:self.border_domain2, :]
        self.data_y = data[self.border_x:self.border_y, self.border_domain1:self.border_domain2, :]
        self.data_stamp = data_stamp[self.border_x:self.border_y, self.border_domain1:self.border_domain2, :]

    def __getitem__(self, index):
        point_i = int(index / (len(self.data_x) - self.seq_len - self.pred_len + 1))
        s_begin = index % (len(self.data_x) - self.seq_len - self.pred_len + 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        data_x = self.data_x.transpose(1, 0, 2)
        data_y = self.data_y.transpose(1, 0, 2)
        data_stamp = self.data_stamp.transpose(1, 0, 2)

        seq_x = data_x[point_i][s_begin:s_end]
        seq_y = data_y[point_i][r_begin:r_end]
        seq_x_mark = data_stamp[point_i][s_begin:s_end]
        seq_y_mark = data_stamp[point_i][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1)*(self.border_domain2 - self.border_domain1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)