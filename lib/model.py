from .layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

import numpy as np



cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class RIPGeo(nn.Module):
	def __init__(self, dim_in, dim_z, dim_med, dim_out, input_dim, embedding_dim, num_heads, collaborative_mlp=True, ):
		super(RIPGeo, self).__init__()

		# # RIPGeo
		# self.att_attribute = SimpleAttention1(temperature=dim_z ** 0.5,
		#                                      d_q_in=dim_in,
		#                                      d_k_in=dim_in,
		#                                      d_v_in=dim_in + 1,
		#                                      d_q_out=dim_z,
		#                                      d_k_out=dim_z,
		#                                      d_v_out=dim_z)

		# if collaborative_mlp:
		#     self.pred = SimpleAttention2(temperature=dim_z ** 0.5,d_q_in=dim_in * 3 + 4,d_k_in=dim_in,d_v_in=2,d_q_out=dim_z,d_k_out=dim_z,d_v_out=2,drop_last_layer=False)
		#
		# else:
		#     self.pred = nn.Sequential(
		#         nn.Linear(dim_z, dim_med),
		#         nn.ReLU(),
		#         nn.Linear(dim_med, dim_out)
		#     )
		#
		# self.collaborative_mlp = collaborative_mlp

		# calculate A
		# self.gamma_1 = nn.Parameter(torch.ones(1, 1))
		# self.gamma_2 = nn.Parameter(torch.ones(1, 1))
		# self.gamma_3 = nn.Parameter(torch.ones(1, 1))
		# self.alpha = nn.Parameter(torch.ones(1, 1))
		# self.beta = nn.Parameter(torch.zeros(1, 1))

		self.embedding_layer = nn.Linear(input_dim, embedding_dim)

		# Define the parameters w1_h and w2_h for each head
		self.w1 = nn.Parameter(torch.randn(embedding_dim, num_heads))
		self.w2 = nn.Parameter(torch.randn(embedding_dim, num_heads))

		self.num_heads = num_heads

		# transform in Graph
		self.w_1 = nn.Linear(dim_in + 3, dim_in + 3)
		self.w_2 = nn.Linear(dim_in + 3, dim_in + 3)
		# self.weight1 = nn.Parameter(torch.randn(dim_in + 2, dim_in + 2))
		# self.weight2 = nn.Parameter(torch.randn(dim_in + 2, dim_in + 2))

		self.pred = nn.Sequential(
			nn.Linear(dim_in + 3, dim_med),
			nn.ReLU(),
			nn.Linear(dim_med, dim_out)
		)
		# self.pred2 = nn.Sequential(
		# 	nn.Linear(2, 10),
		# 	nn.ReLU(),
		# 	nn.Linear(10, 2)
		# )
	def compute_edge_prob(self, embedded_nodes, head):
		# Similarity function s(·, ·), you can choose different ones
		# similarity = torch.matmul(embedded_nodes, self.w1[:, head]) * torch.matmul(embedded_nodes, self.w2[:, head])
		similarity = F.cosine_similarity((embedded_nodes * self.w1[:, head]).unsqueeze(1),
										 (embedded_nodes * self.w2[:, head]).unsqueeze(0), dim=2)


		# Compute edge probability αuv
		alpha = torch.sigmoid(similarity)

		return alpha

	def get_community_labels(self, data, num_clusters):
		kmeans = KMeans(n_clusters=num_clusters)
		labels = kmeans.fit_predict(data)
		return labels

	def generate_adjacency_matrix_based_on_community(self, community_labels, k, X):
		n = len(community_labels)
		adjacency_matrix = np.zeros((n, n), dtype=int)

		for i in range(n):
			same_community = torch.nonzero(community_labels == community_labels[i]).squeeze()
			distances = torch.cdist(X[same_community], X[i].unsqueeze(0))
			_, nearest_indices = distances.topk(k + 1, largest=False, sorted=True)
			nearest_indices = nearest_indices.squeeze()
			nearest_indices = [ind for ind in nearest_indices if ind != i.item()]
			for idx in nearest_indices:
				adjacency_matrix[i, same_community[idx]] = 1

		return adjacency_matrix



	def forward(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay):
		lm_community_labels = self.get_community_labels(lm_X.numpy(), num_clusters=5)
		tg_community_labels = self.get_community_labels(tg_X.numpy(), num_clusters=5)

		# 仅选择特定社区的数据
		selected_lm_indices = (lm_community_labels == selected_community_label)  # 选择特定的社区标签
		selected_tg_indices = (tg_community_labels == selected_community_label)  # 选择特定的社区标签

		# 使用选择的数据进行模型输入
		lm_X = lm_X[selected_lm_indices]
		lm_Y = lm_Y[selected_lm_indices]
		tg_X = tg_X[selected_tg_indices]
		tg_Y = tg_Y[selected_tg_indices]
		lm_delay = lm_delay[selected_lm_indices]
		tg_delay = tg_delay[selected_tg_indices]

		self.lm_X = lm_X  # 将lm_X设置为模型属性，以便在邻接矩阵生成方法中使用

		# Generate adjacency matrix based on community labels
		k = 5  # 最近邻的数量，你也可以根据需要进行调整
		adj_lm = self.generate_adjacency_matrix_based_on_community(lm_community_labels, k)
		adj_tg = self.generate_adjacency_matrix_based_on_community(tg_community_labels, k)

		# 生成社区邻接矩阵
		k = 5  # 最近邻的数量，你也可以根据需要进行调整
		adj_lm = self.generate_adjacency_matrix_based_on_community(lm_community_labels, k)
		adj_tg = self.generate_adjacency_matrix_based_on_community(tg_community_labels, k)

		# 更新 y_pred
		N1 = lm_X.size(0)
		N2 = lm_Y.size(0)
		if N1 < 5:
			k = N1
			adj_lm = torch.ones(adj_lm.shape)
		else:
			k = 5
		# 你可能需要根据需要的操作来更新 adj_lm

		if N1 > 5:
			top_values, top_indices = torch.topk(adj_lm, k, dim=1)
			output_matrix = torch.zeros_like(adj_lm)
			output_matrix.scatter_(1, top_indices, top_values)
			adj_lm = output_matrix

		adj_lm = adj_lm.cuda()
		y_pred = adj_lm @ lm_Y * (1 / k)

		# 更新 adj_teacher
		adj_teacher = torch.eye(N1 + N2)
		adj_teacher[adj_teacher == 1] = -1
		adj_teacher += 1

		adj_teacher[:N1, :N1] = Tensor(adj_lm)
		adj_teacher[-N2:, -N2:] = 0

		# 更新 adj
		# 这里可能需要根据你的计算更新 adj

		# 邻接矩阵学习
		# N1 = lm_X.size(0)
		# N2 = lm_Y.size(0)
		#
		lm_feature_ = torch.cat((lm_X, lm_delay.unsqueeze(-1)), dim=1)
		tg_feature_ = torch.cat((tg_X, tg_delay.unsqueeze(-1)), dim=1)
		adj_learn_feature = torch.cat((lm_feature_, tg_feature_), dim=0)

		embedded_nodes = self.embedding_layer(adj_learn_feature)
		#
		#
		# Compute edge probabilities
		alpha = torch.zeros((N1 + N2, N1 + N2, self.num_heads))
		for h in range(self.num_heads):
			alpha[:, :, h] = self.compute_edge_prob(embedded_nodes, h)

		adj = torch.mean(alpha, dim=-1)

		#
		# # 魔改0： 邻接矩阵指导
		# adj_teacher = torch.eye(N1 + N2)
		# adj_teacher[adj_teacher == 1] = -1
		# adj_teacher += 1
		#
		# if N1 > 5:
		# 	k = 5
		# else:
		# 	k = N1 - 1
		# adj_lm = self.nearest_neighbors_matrix(lm_X.to('cpu').numpy(), k)
		#
		# adj_teacher[:N1, :N1] = Tensor(adj_lm)
		# adj_teacher[-N2:, -N2:] = 0

		# 原始GCN版本
		# lm_feature = torch.cat((lm_X, lm_delay.unsqueeze(-1), lm_Y), dim=1)
		# tg_feature = torch.cat((tg_X, tg_delay.unsqueeze(-1), torch.zeros(N2, 2).cuda()), dim=1)
		# all_feature = torch.cat((lm_feature, tg_feature), dim=0)
		#
		# degree_0 = torch.sum(adj, dim=1)
		# degree_reverse_0 = 1.0 / degree_0
		# degree_matrix_reverse_0 = torch.diag(degree_reverse_0)
		# degree_mul_adj_0 = degree_matrix_reverse_0 @ adj
		#
		# degree_mul_adj_0 = degree_mul_adj_0.cuda()
		# all_feature = self.w_1(degree_mul_adj_0 @ all_feature)
		# all_feature = F.relu(all_feature)
		#
		# all_feature = self.w_1(degree_mul_adj_0 @ all_feature)
		# all_feature = F.relu(all_feature)
		#
		# y_pred = self.pred(all_feature[N1:N1 + N2, :])

		# 魔改1：removeGCN版本: 将adj中的数据全部设置为1
		# adj2 = adj[N1:N1 + N2, :N1]
		# if N1 < 5:
		# 	k = N1
		# 	adj2 = torch.ones(adj2.shape)
		# else:
		# 	k = 5
		# 	adj2 = self.top_k_values_to_1(adj2, k=k)

		# # 魔改2：removeGCN版本: 将adj中的参数全部保留最大的几个
		# adj2 = adj[N1:N1 + N2, :N1]
		# if N1 < 5:
		# 	k = N1
		# 	adj2 = torch.ones(adj2.shape)
		# else:
		# 	k = 5
		# 	adj2 = self.top_k_values_to_1(adj2, k=k)
		#
		# if N1 > 5:
		# 	top_values, top_indices = torch.topk(adj2, k, dim=1)
		#
		# 	# 创建一个与输入矩阵相同大小的零矩阵
		# 	output_matrix = torch.zeros_like(adj2)
		#
		# 	# 将对应的索引位置置为1
		# 	output_matrix.scatter_(1, top_indices, top_values)
		# 	adj2 = output_matrix
		#
		# adj2 = adj2.cuda()
		# y_pred = adj2 @ lm_Y * (1 / k)


		# # 魔改3： 注意力+直接计算
		# _, attribute_score_0 = self.att_attribute(tg_X, lm_X, lm_feature_)
		# adj2 = attribute_score_0
		# if N1 < 5:
		# 	k = N1
		# 	adj2 = torch.ones(adj2.shape)
		# else:
		# 	k = 5
		# 	adj2 = self.top_k_values_to_1(adj2, k=k)
		#
		# adj2 = adj2.cuda()
		# # y_pred = self.pred2(adj2 @ lm_Y)  # * (1 / k)
		# y_pred = adj2 @ lm_Y * (1 / k)
		return y_pred, adj, adj_teacher






