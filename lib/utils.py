from __future__ import print_function
import numpy as np
import torch
import warnings
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import copy

warnings.filterwarnings(action='once')


class DataPerturb:
    def __init__(self, eta=1):
        self.eta = eta
        self.loss = torch.nn.MSELoss(reduction='sum')

    def perturb(self, model, data):
        # original
        lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay = data

        # obtain new graph representation
        _, ori_graph_feature = model(lm_X, lm_Y, tg_X,
                                     tg_Y, lm_delay,
                                     tg_delay)

        # add Gaussian data perturb
        new_lm_X, new_lm_Y, new_tg_X, new_tg_Y, new_lm_delay, new_tg_delay = lm_X.clone(), lm_Y.clone(), \
                                                                             tg_X.clone(), tg_Y.clone(), \
                                                                             lm_delay.clone(), tg_delay.clone()
        new_lm_X[:, -16:] += self.eta * torch.normal(0, torch.ones_like(new_lm_X[:, -16:]) * new_lm_X[:, -16:]).cuda()
        new_tg_X[:, -16:] += self.eta * torch.normal(0, torch.ones_like(new_tg_X[:, -16:]) * new_tg_X[:, -16:]).cuda()
        new_lm_delay += self.eta * torch.normal(0, torch.ones_like(new_lm_delay) * new_lm_delay).cuda()
        new_tg_delay += self.eta * torch.normal(0, torch.ones_like(new_tg_delay) * new_tg_delay).cuda()

        # obtain new graph representation
        _, new_graph_feature = model(new_lm_X, new_lm_Y, new_tg_X,
                                     new_tg_Y, new_lm_delay,
                                     new_tg_delay)

        data_loss = self.loss(ori_graph_feature, new_graph_feature)
        return data_loss


class ParaPerturb:
    def __init__(self, zeta=1, mu=0.4, step=3):
        self.zeta = zeta
        self.mu = mu
        self.step = step
        self.ori_model = None
        self.model = None
        self.vice_model = None
        self.emb_backup = {}

        self.loss = torch.nn.MSELoss(reduction='mean')

    def load_model(self, model):
        self.ori_model = model
        self.model = copy.deepcopy(model)
        self.vice_model = copy.deepcopy(model)
        for name, param in self.model.named_parameters():    # 返回name和param
            self.emb_backup[name] = param.data.clone()

    def random_attack(self, data):
        lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay = data

        _, ori_graph_feature = self.model(lm_X, lm_Y, tg_X,
                                          tg_Y, lm_delay,
                                          tg_delay)

        # perturb with Gaussian noise
        for (_, adv_param), (_, param) in zip(self.vice_model.named_parameters(), self.model.named_parameters()):
            adv_param.data = param.data + self.zeta * torch.normal(0, torch.abs(torch.ones_like(param.data) * param.data)).cuda()     # 对adv_param.data直接赋值是会直接改变这个模型参数的

        _, new_graph_feature = self.vice_model(lm_X, lm_Y, tg_X,
                                               tg_Y, lm_delay,
                                               tg_delay)

        random_para_loss = self.loss(new_graph_feature, ori_graph_feature)
        random_para_loss.backward()  # 在这里梯度反传后，参数中的grad才不为None

    def attack(self):
        for (_, adv_param), (name, param) in zip(self.vice_model.named_parameters(), self.model.named_parameters()):
            if adv_param.grad is not None:
                norm = torch.norm(adv_param.grad)
            else:
                norm = 0
            if norm != 0 and not torch.isnan(norm):
                r_at = self.mu * adv_param.grad / norm  # 归一化
                adv_param.data.add_(r_at)     # 朝着继续拉大的方向改变参数
                param.data = self.project(name, adv_param.data)   # self.model 在这时就相当于是self.vice_model的限制版本，可以想象成让self.vice_model一次迈的步子不要太大了

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > 2 * self.zeta * torch.norm(self.emb_backup[param_name]):
            r = 2 * self.zeta * torch.norm(self.emb_backup[param_name]) * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def attack_loss(self, data, is_last_one=False):
        lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay = data

        if not is_last_one:
            _, ori_graph_feature = self.model(lm_X, lm_Y, tg_X,
                                              tg_Y, lm_delay,
                                              tg_delay)

            _, new_graph_feature = self.vice_model(lm_X, lm_Y, tg_X,
                                                   tg_Y, lm_delay,
                                                   tg_delay)
        else:
            _, ori_graph_feature = self.ori_model(lm_X, lm_Y, tg_X,
                                                  tg_Y, lm_delay,
                                                  tg_delay)

            _, new_graph_feature = self.vice_model(lm_X, lm_Y, tg_X,
                                                   tg_Y, lm_delay,
                                                   tg_delay)
        attack_loss = self.loss(new_graph_feature, ori_graph_feature)
        return attack_loss

    def perturb(self, model, data):
        # initialize model, using deepcopy to prevent model from losing gradient of previous data perturbation
        self.load_model(model)

        # initialize perturbation with Gaussian noise
        self.random_attack(data)

        # inner maximization
        for t in range(self.step):
            self.attack()
            self.vice_model.zero_grad()  # 每次反向传播后，没有更新参数就改变了梯度哦
            self.model.zero_grad()
            attack_loss = self.attack_loss(data, is_last_one=(t == self.step - 1))
            # print(t, attack_loss)
            if t != (self.step - 1):
                attack_loss.backward()
            else:
                pass
        return attack_loss


class MaxMinLogRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min)


class MaxMinRTTScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data_o = np.array(data)
        # data_o = np.log(data_o + 1)
        return (data_o - self.min) / (self.max - self.min)


class MaxMinLogScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def transform(self, data):
        data[data != 0] = -np.log(data[data != 0] + 1)
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data[data != 0] = (data[data != 0] - min) / (max - min)
        return data

    def inverse_transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        data = data * (max - min) + min
        return np.exp(data)


class MaxMinScaler():
    def __init__(self):
        self.min = 0.
        self.max = 1.

    def fit(self, data):
        data_o = np.array(data)
        self.max = data_o.max()
        self.min = data_o.min()

    def transform(self, data):
        max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / (max - min)

    def inverse_transform(self, data):
        # max = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        # min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return data * (self.max - self.min) + self.min

def restore_original_data(lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay,  g, center=0, normal=2):
    if normal == 2:

        lm_X = lm_X * (g["x_max"] - g["x_min"] + 1e-12) + g["x_min"]
        tg_X = tg_X * (g["x_max"] - g["x_min"] + 1e-12) + g["x_min"]

        lm_Y = lm_Y * (g["y_max"] - g["y_min"] + 1e-12) + g["y_min"]
        tg_Y = tg_Y * (g["y_max"] - g["y_min"] + 1e-12) + g["y_min"]

        center = center * (g["y_max"] - g["y_min"] + 1e-12) + g["y_min"]

        lm_delay = np.exp(lm_delay * (g["delay_max"] - g["delay_min"] + 1e-12) + g["delay_min"])
        tg_delay = np.exp(tg_delay * (g["delay_max"] - g["delay_min"] + 1e-12) + g["delay_min"])


    # elif normal == 1:
    #     for g in graphs:
    #
    #         g["lm_X"] = g["lm_X"].squeeze(axis=1) * (X_max - X_min + 1e-12) + X_min
    #         g["tg_X"] = g["tg_X"].squeeze(axis=1) * (X_max - X_min + 1e-12) + X_min
    #
    #         g["lm_Y"] = g["lm_Y"].squeeze(axis=1) * (Y_max - Y_min + 1e-12) + Y_min
    #         g["tg_Y"] = g["tg_Y"].squeeze(axis=1) * (Y_max - Y_min + 1e-12) + Y_min
    #         g["center"] = g["center"] * (Y_max - Y_min + 1e-12) + Y_min
    #
    #         g["lm_delay"] = np.exp(g["lm_delay"].squeeze(axis=1) * (delay_max - delay_min + 1e-12) + delay_min)
    #         g["tg_delay"] = np.exp(g["tg_delay"].squeeze(axis=1) * (delay_max - delay_min + 1e-12) + delay_min)
    #
    #         # g["y_max"], g["y_min"] = [1, 1], [0, 0]

    return lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, center

def graph_normal(graphs, normal=2):
    if normal == 2:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            # if Y.max(axis=0)[0] == Y.min(axis=0)[0] and Y.max(axis=0)[1] == Y.min(axis=0)[1]:
            #     print("*"*10)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)
            g["x_max"], g["x_min"] = X.max(axis=0), X.min(axis=0)
            g["delay_max"], g["delay_min"] = delay.max(), delay.min()
            # g["y_max"], g["y_min"] = Y.max(axis=0), Y.min(axis=0)
            # if g["y_max"][0] == g["y_min"][0] and g["y_max"][1] == g["y_min"][1]:
            #     print("*"*10)

    elif normal == 1:
        for g in graphs:
            X = np.concatenate((g["lm_X"], g["tg_X"]), axis=0).squeeze(axis=1)  # [n, 30]

            g["lm_X"] = (g["lm_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
            g["tg_X"] = (g["tg_X"].squeeze(axis=1) - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)

            Y = np.concatenate((g["lm_Y"], g["tg_Y"]), axis=0).squeeze(axis=1)
            g["lm_Y"] = (g["lm_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["tg_Y"] = (g["tg_Y"].squeeze(axis=1) - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)
            g["center"] = (g["center"] - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0) + 1e-12)

            delay = np.concatenate((g["lm_delay"], g["tg_delay"]), axis=0).squeeze(axis=1)

            g["lm_delay"] = (np.log(g["lm_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)
            g["tg_delay"] = (np.log(g["tg_delay"].squeeze(axis=1)) - np.log(delay.min())) / (
                    np.log(delay.max()) - np.log(delay.min()) + 1e-12)

            g["y_max"], g["y_min"] = [1, 1], [0, 0]

    return graphs


def get_data_generator(opt, data_train, data_test, normal=2):
    # load data
    data_train = data_train[np.array([graph["exist"] for graph in data_train])]
    data_test = data_test[np.array([graph["exist"] for graph in data_test])]

    data_train, data_test = graph_normal(data_train, normal=normal), graph_normal(data_test, normal=normal)

    random.seed(opt.seed)
    random.shuffle(data_train)
    random.seed(opt.seed)
    random.shuffle(data_test)

    return data_train, data_test


def dis_loss(y, y_pred, max, min):
    y[:, 0] = y[:, 0] * (max[0] - min[0] +1e-12 )
    y[:, 1] = y[:, 1] * (max[1] - min[1]+1e-12 )
    y_pred[:, 0] = y_pred[:, 0] * (max[0] - min[0]+ 1e-12 )
    y_pred[:, 1] = y_pred[:, 1] * (max[1] - min[1] +1e-12 )

    distance = torch.sqrt((((y - y_pred) * 100) * ((y - y_pred) * 100)).sum(dim=1))
    return distance

# save checkpoint of model
def save_cpt(model, optim, epoch, save_path):
    """
    save checkpoint, for inference/re-training
    :return:
    """
    model.eval()
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },
        save_path
    )

def get_adjancy(func, delay, hop, nodes):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    hops = []
    delays = []
    x1 = []
    x2 = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            delays.append(delay[i, j])
            hops.append(hop[i, j])
            x1.append(nodes[i].cpu().detach().numpy())
            x2.append(nodes[j].cpu().detach().numpy())
    dis = func(Tensor(delays), Tensor(hops), Tensor(x1), Tensor(x2))
    A = torch.zeros_like(delay)
    index = 0
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            A[i, j] = dis[index]
            index += 1
    return A


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            # nn.init.constant_(m.bias, val=0)


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))


def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))


def draw_cdf(ds_sort):
    last, index = min(ds_sort), 0
    x = []
    y = []
    while index < len(ds_sort):
        x.append([last, ds_sort[index]])
        y.append([index / len(ds_sort), index / len(ds_sort)])

        if index < len(ds_sort):
            last = ds_sort[index]
        index += 1
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(x).reshape(-1, 1).squeeze(),
             np.array(y).reshape(-1, 1).squeeze(),
             c='k',
             lw=2,
             ls='-')
    plt.xlabel('Geolocation Error(km)')
    plt.ylabel('Cumulative Probability')
    plt.grid()
    plt.show()
