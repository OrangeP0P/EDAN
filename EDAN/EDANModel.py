import numpy as np
import torch
import torch.nn as nn
from MMD import mmd_rbf
import torch.utils.data
from torch.nn import functional as F

class EDAN(nn.Module):
    def __init__(self, act_func):
        super(EDAN, self).__init__()
        self.F1 = 8
        self.D = 2
        self.F2 = 16
        self.channel = 118
        self.T = 350
        self.kernel_size = 64
        self.ELE_feature = 80
        self.ALL_feature = 112
        self.ET = torch.from_numpy(np.load('ET.npy', allow_pickle=True)).cuda()  # 读取权重

        self.Sequence1 = nn.Sequential()
        self.Sequence1.add_module('A-Conv1', nn.Conv2d(1, self.F1, (1, 25), stride=(1, 1)))
        self.Sequence1.add_module('A-Norm1', nn.BatchNorm2d(self.F1, False))
        self.Sequence1.add_module('A-ELU1', nn.ReLU())
        self.Sequence1.add_module('A-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence1.add_module('A-Drop1', nn.Dropout(p=0.25))

        self.Sequence2 = nn.Sequential()
        self.Sequence2.add_module('B-Conv1', nn.Conv2d(self.F1, self.F1 * 2, (3, 1), stride=(1, 1)))
        self.Sequence2.add_module('B-ELU1', nn.ReLU())
        self.Sequence2.add_module('B-Drop1', nn.Dropout(p=0.25))

        self.Sequence3 = nn.Sequential()
        self.Sequence3.add_module('C-Conv1', nn.Conv2d(self.F1*2, self.F1*2, (1, 10), stride=(1, 1)))
        self.Sequence3.add_module('C-Norm1', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence3.add_module('C-ELU1', nn.ReLU())
        self.Sequence3.add_module('C-AVGPool1', nn.AvgPool2d((1, 5)))
        self.Sequence3.add_module('C-Drop1', nn.Dropout(p=0.25))

        self.SFC1 = nn.Sequential()
        self.SFC1.add_module('SFC1', nn.Linear(11, 11))
        self.SFC1.add_module('SFC1-Norm1', nn.BatchNorm2d(16))

        self.SFC2 = nn.Sequential()
        self.SFC2.add_module('SFC2', nn.Linear(11, 11))
        self.SFC2.add_module('SFC2-Norm1', nn.BatchNorm2d(16))

        self.Sequence4 = nn.Sequential()
        self.Sequence4.add_module('S-Conv2', nn.Conv2d(self.F1*2, self.F1*2, (118, 1), stride=(1, 1)))
        self.Sequence4.add_module('S-Norm2', nn.BatchNorm2d(self.F1*2, False))
        self.Sequence4.add_module('S-ELU1', nn.ReLU())
        self.Sequence4.add_module('S-Drop1', nn.Dropout(p=0.25))

        self.FC1 = nn.Sequential()
        self.FC1.add_module('E_FC1', nn.Linear(176, 256))
        self.FC1.add_module('E-FC-Norm2', nn.BatchNorm1d(256))

        self.F_FC1 = nn.Sequential()
        self.F_FC1.add_module('F_FC1', nn.Linear(256, 64))
        self.F_FC1.add_module('F-Norm1', nn.BatchNorm1d(64))
        self.F_FC1.add_module('F_FC2', nn.Linear(64, 2))

    def forward(self, source, target):
        glob_loss = 0
        intra_source_ele_loss = 0
        intra_target_ele_loss = 0
        intra_ele_loss = 0
        inter_ele_loss = 0

        source_all = self.Sequence1(source)
        source_all = F.pad(source_all, (1, 1, 2, 0))
        source_all = self.Sequence2(source_all)
        source_all = self.Sequence3(source_all)
        source_ele_1 = self.SFC1(source_all)

        # 源域用户内电极损失 第一层
        _s0, _s1, _s2 = source_ele_1.shape[:3]
        SE1 = source_ele_1.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)

        _s = SE1 - SE1.unsqueeze(1)
        _s = _s @ _s.transpose(-1, -2)
        _ms1 = _s.mean(dim=-1)
        _ms2 = _ms1.mean(dim=-1)
        _ms3 = _ms2 * self.ET
        _ms4 = _ms3.sum()
        intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)


        source_ele_2 = self.SFC2(source_ele_1)


        # 源域用户内电极损失 第二层
        _s0, _, _s2 = source_ele_2.shape[:3]
        SE2 = source_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)
        _s = SE2 - SE2.unsqueeze(1)
        _s = _s @ _s.transpose(-1, -2)
        _ms1 = _s.mean(dim=-1)
        _ms2 = _ms1.mean(dim=-1)
        _ms3 = _ms2 * self.ET
        _ms4 = _ms3.sum()

        intra_source_ele_loss += _ms4 / (_s2 * _s2) / (_s1 * _s1)

        source_all = self.Sequence4(source_ele_2)

        if self.training:
            target_all = self.Sequence1(target)
            target_all = F.pad(target_all, (1, 1, 2, 0))
            target_all = self.Sequence2(target_all)
            target_all = self.Sequence3(target_all)
            target_ele_1 = self.SFC1(target_all)  # 3D全连接层1

            # 目标域用户内电极损失 第一层
            _t0, _t1, _t2 = target_ele_1.shape[:3]
            TE1 = target_ele_1.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)

            _t = TE1 - TE1.unsqueeze(1)
            _t = _t @ _t.transpose(-1, -2)
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)
            _mt3 = _mt2 * self.ET
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            target_ele_2 = self.SFC2(target_ele_1)

            # 目标域用户内电极损失 第二层
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_t2, _t0, -1)
            _t = TE2 - TE2.unsqueeze(1)
            _t = _t @ _t.transpose(-1, -2)
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)
            _mt3 = _mt2 * self.ET
            _mt4 = _mt3.sum()
            intra_target_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            intra_ele_loss += intra_source_ele_loss + intra_target_ele_loss

            # 源域、目标域用户间电极损失
            _t0, _t1, _t2 = target_ele_2.shape[:3]
            TE2 = target_ele_2.permute(2, 0, 1, 3).reshape(_s2, _s0, -1)
            _t = TE2 - SE2.unsqueeze(1)
            _t = _t @ _t.transpose(-1, -2)
            _mt1 = _t.mean(dim=-1)
            _mt2 = _mt1.mean(dim=-1)
            _mt3 = _mt2 * self.ET
            _mt4 = _mt3.sum()
            inter_ele_loss += _mt4 / (_t2 * _t2) / (_t1 * _t1)

            target_all = self.Sequence4(target_ele_2)

            # 拉伸成条形处理
            s0, s1, s2, s3 = target_all.shape[:4]
            target_all = target_all.reshape(s0, s1 * s3)
            source_all = source_all.reshape(s0, s1 * s3)

            target_all = self.FC1(target_all)
            source_all = self.FC1(source_all)

            glob_loss += mmd_rbf(source_all, target_all, kernel_mul=5.0, kernel_num=10, fix_sigma=None)

            output = self.F_FC1(source_all)

        return output, glob_loss, intra_ele_loss, inter_ele_loss

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss