#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(n_emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x



