from .layers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class MemoryUnit(nn.Module):
	def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
		super(MemoryUnit, self).__init__()
		self.mem_dim = mem_dim
		self.fea_dim = fea_dim
		self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
		#         print("memory shape", self.weight.shape)
		self.bias = None
		self.shrink_thres = shrink_thres
		# self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input):
		att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
		att_weight = F.softmax(att_weight, dim=1)  # TxM
		# ReLU based shrinkage, hard shrinkage for positive value
		if (self.shrink_thres > 0):
			att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
			# att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
			# normalize???
			att_weight = F.normalize(att_weight, p=1, dim=1)
		# att_weight = F.softmax(att_weight, dim=1)
		# att_weight = self.hard_sparse_shrink_opt(att_weight)
		mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
		output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
		return {'output': output, 'att': att_weight}  # output, att_weight

	def extra_repr(self):
		return 'mem_dim={}, fea_dim={}'.format(
			self.mem_dim, self.fea_dim is not None
		)

	def hard_shrink_relu(self, input, lambd=0, epsilon=1e-12):
		output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
		return output


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
	def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
		super(MemModule, self).__init__()
		self.mem_dim = mem_dim
		self.fea_dim = fea_dim
		self.shrink_thres = shrink_thres
		self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

	def forward(self, input):
		# s = input.data.shape
		# l = len(s)  # [batch_size, ch, time_length, imh, imw]  [bs, feature_size]
		#
		# if l == 3:
		# 	x = input.permute(0, 2, 1)
		# elif l == 4:
		# 	x = input.permute(0, 2, 3, 1)
		# elif l == 5:
		# 	x = input.permute(0, 2, 3, 4, 1)  # [batch_size, time, imh, imw, ch]
		# else:
		# 	x = []
		# 	print('wrong feature map size')
		# x = x.contiguous()
		# x = x.view(-1, s[1])  # [batch_size * time * imh * imw, ch]
		#

		y_and = self.memory(input)
		#
		y = y_and['output']
		att = y_and['att']

		# if l == 3:
		# 	y = y.view(s[0], s[2], s[1])
		# 	y = y.permute(0, 2, 1)
		# 	att = att.view(s[0], s[2], self.mem_dim)
		# 	att = att.permute(0, 2, 1)
		# elif l == 4:
		# 	y = y.view(s[0], s[2], s[3], s[1])
		# 	y = y.permute(0, 3, 1, 2)
		# 	att = att.view(s[0], s[2], s[3], self.mem_dim)
		# 	att = att.permute(0, 3, 1, 2)
		# elif l == 5:
		# 	y = y.view(s[0], s[2], s[3], s[4], s[1])
		# 	y = y.permute(0, 4, 1, 2, 3)
		# 	att = att.view(s[0], s[2], s[3], s[4],
		# 	               self.mem_dim)  # [batch_size, time_length, imh, imw, memory_dimension]
		# 	att = att.permute(0, 4, 1, 2, 3)  # [batch_size, memory_dimension, time_length, imh, imw]
		# else:
		# 	y = x
		# 	att = att
		# 	print('wrong feature map size')
		return {'output': y, 'att': att}


# relu based hard shrinkage function, only works for positive values


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


class RIPGeoContrast(nn.Module):
	def __init__(self, dim, dim_z, num_classes=2):
		super(RIPGeoContrast, self).__init__()
		self.in_channels = dim
		self.conv1 = nn.Conv1d(1, dim, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(dim)
		self.layer1 = self._make_layer(dim, dim_z // 8, 2, stride=1)
		self.layer2 = self._make_layer(dim_z // 8, dim_z // 4, 2, stride=2)
		self.layer3 = self._make_layer(dim_z // 4, dim_z // 2, 2, stride=2)
		self.layer4 = self._make_layer(dim_z // 2, dim_z, 2, stride=2)
		self.memory = MemModule(mem_dim=30, fea_dim=dim_z, shrink_thres=0.0025)  # 这里的参数：mem_dim和shrink_thres可以写到args中

	# self.linear = nn.Linear(512, num_classes)

	def _make_layer(self, in_channels, out_channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(BasicBlock(in_channels, out_channels, kernel_size=3, stride=stride))
			in_channels = out_channels
		return nn.Sequential(*layers)

	def compute_distance_and_knn(self, dis, k):
		# 计算每两个节点之间的距离
		dists = torch.cdist(dis, dis)

		# 初始化一个全零的矩阵
		knn = torch.zeros_like(dists)

		# 对于每一行（每个节点）
		for i in range(dists.shape[0]):
			# 获取最近的k个节点的索引
			knn_indices = dists[i].topk(k, largest=False).indices
			# 将最近的k个节点的关系表示为1
			knn[i][knn_indices] = 1

		return knn

	def keep_top_k_elements(self, adj, k):
		result = adj.clone()

		mask = torch.zeros_like(result)
		for i in range(result.size(0)):
			# 找到每一行中第 k 大的元素
			_, indices = torch.topk(result[i], k, largest=False)

			# 将小于第 k 大元素的值置为零
			mask[i, indices] = 1

		result = result * mask

		return result

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
		out = out.view(out.size(0), -1)  # out --memory--> out_hat    [M * out.shape[1]]

		out = self.memory(out) # 加入memory

		feature = out['output']

		feature_lm = feature[:lm_Y.size(0), :]
		feature_tg = feature[lm_Y.size(0):, :]

		try:
			# 这里可以改为注意力模型 TODO
			adj = torch.mm(feature_tg, feature_lm.T)
		except:
			print(feature_lm.shape, feature_tg.shape)

		# Changes:只保留最大的k个数据（一半）
		if lm_Y.size(0) < 4:
			k_keep = lm_Y.size(0)
		else:
			k_keep = lm_Y.size(0) // 2
		# print(adj.shape,k_keep)
		adj = self.keep_top_k_elements(adj, k_keep)
		adj = F.normalize(adj, p=1, dim=1)
		pred = torch.mm(adj, lm_Y)
		# out = self.linear(feature)

		# 根据Y计算mask
		if n_ips < 5:
			k = n_ips
		elif n_ips < 30:
			k = int(0.4 * n_ips)
		else:
			k = int(0.1 * n_ips)

		mask = self.compute_distance_and_knn(Y, k)

		return feature, pred, mask
