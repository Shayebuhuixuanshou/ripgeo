
import community  # 导入Louvain社区检测算法库
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RIPGeoCommunity(nn.Module):
    def __init__(self, dim, dim_z, num_classes=2):
        super(RIPGeoCommunity, self).__init__()
        self.in_channels = dim
        self.conv1 = nn.Conv1d(1, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.layer1 = self._make_layer(dim, dim_z // 8, 2, stride=1)
        self.layer2 = self._make_layer(dim_z // 8, dim_z // 4, 2, stride=2)
        self.layer3 = self._make_layer(dim_z // 4, dim_z // 2, 2, stride=2)
        self.layer4 = self._make_layer(dim_z // 2, dim_z, 2, stride=2)

    # self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_channels, out_channels, kernel_size=3, stride=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    #社区生成算法
    def generate_adjacency_matrix(self, X, n):#n是社区数量
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))

        # 使用节点特征间的距离生成边权重
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    G.add_edge(i, j, weight=torch.dist(X[i], X[j]))

        partition = community.best_partition(G, resolution=n)  # 使用 Louvain 算法检测社区

        # 根据社区信息生成邻接矩阵
        adjacency_matrix = torch.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                if partition[i] == partition[j]:
                    adjacency_matrix[i][j] = 1

        return adjacency_matrix



    def forward(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay):

        n_ips = lm_Y.size(0) + tg_Y.size(0)

        # 特征向量拼接
        lm_feature_ = torch.cat((lm_X, lm_delay.unsqueeze(-1)), dim=1)
        tg_feature_ = torch.cat((tg_X, tg_delay.unsqueeze(-1)), dim=1)

        X = torch.cat((lm_feature_, tg_feature_), dim=0)
        Y = torch.cat((lm_Y, tg_Y), dim=0)

        out = self.conv1(X.unsqueeze(1))
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)

        feature = out.clone()

        feature_lm = out[:lm_Y.size(0), :]
        feature_tg = out[lm_Y.size(0):, :]

        adj = torch.mm(feature_tg, feature_lm.T)
        adj = F.normalize(adj, p=1, dim=1)
        pred = torch.mm(adj, lm_Y)
        # out = self.linear(feature)

        # 根据Y计算mask
        if n_ips < 5:
            n = n_ips
        elif n_ips < 30:
            n = int(0.4 * n_ips)
        else:
            n = int(0.1 * n_ips)

        mask = self.generate_adjacency_matrix(X, n)#修改

        return feature, pred, mask
