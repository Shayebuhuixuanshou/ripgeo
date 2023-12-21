# -*- coding: utf-8 -*-
import torch.nn

from lib.utils import *
import argparse
import numpy as np
import random, os
from lib.model import *
# import wandb
import datetime
from lib.loss import SupConLoss
from lib.model_community import *
import copy


# from thop import profile

description = "改进的回退机制，harved_epoch=20，初始lr=0.0001，early_stop_epoch=200"

parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=1024, help='manual seed')
parser.add_argument('--description', type=str, default=description)
parser.add_argument('--model_name', type=str, default='RIPGeoCommunity-调试程序')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--loss_contrast_ratio', type=float, default=1)
parser.add_argument('--loss_mse_ratio', type=float, default=0)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--harved_epoch', type=int, default=20)
parser.add_argument('--early_stop_epoch', type=int, default=200)
parser.add_argument('--saved_epoch', type=int, default=100)

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")

parser.add_argument('--dim_z', type=int, default=128)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--epoch_threshold', type=int, default=5)

opt = parser.parse_args()

if opt.seed:
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # 补充随机种子
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
# torch.backends.cudnn.deterministic = True  # 加上运行会很慢
torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
print("Dataset: ", opt.dataset)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

'''load data'''
train_data = np.load("datasets/{}/Clustering_s1234_lm70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("datasets/{}/Clustering_s1234_lm70_test.npz".format(opt.dataset),
                    allow_pickle=True)

train_data, test_data = train_data["data"], test_data["data"]
print("data loaded.")

'''initiate model'''
model = RIPGeoCommunity(dim=opt.dim_in + 1, dim_z=opt.dim_z)  # 修改
loss_fun = SupConLoss(temperature=0.07)

print(opt)
model.apply(init_network_weights)
if cuda:
    model.cuda()

'''initiate criteria and optimizer'''
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

'''Changes：加入num_back,引入回退机制'''
num_back = 0

if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data, normal=2)

    log_path = f"asset/log/{opt.model_name}"
    model_path = f"asset/model/{opt.model_name}"

    for path in [log_path, model_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    # 获取当前时间
    current_time = datetime.datetime.now()

    f = open(
        f"{log_path}/{opt.dataset}_{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-{current_time.minute}.txt",
        'a')
    f.write(f"\n*********{opt.model_name}_{opt.dataset}*********\n")
    f.write("dim=" + str(opt.dim) + ", ")
    f.write("dim_z=" + str(opt.dim_z) + ", ")
    f.write("loss_contrast_ratio=" + str(opt.loss_contrast_ratio) + ", ")
    f.write("loss_mse_ratio=" + str(opt.loss_mse_ratio) + ", ")
    f.write("early_stop_epoch=" + str(opt.early_stop_epoch) + ", ")
    f.write("harved_epoch=" + str(opt.harved_epoch) + ", ")
    f.write("saved_epoch=" + str(opt.saved_epoch) + ", ")
    f.write("lr=" + str(opt.lr) + ", ")
    f.write("model_name=" + opt.model_name + ", ")  # description
    f.write("seed=" + str(opt.seed) + ",")
    f.write("\n")

    f.write("description=" + str(opt.description))
    f.write("\n")

    f.close()

    # train
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0

    epoch_threshold = opt.epoch_threshold
    mse_loss_adj = torch.nn.MSELoss()
    lab = 0
    for epoch in range(2000):
        print("epoch {}.    ".format(epoch))
        beta = min([(epoch * 1.) / max([100, 1.]), 1.])
        total_loss, total_mae, train_num, total_data_perturb_loss = 0, 0, 0, 0
        model.train()
        count = 0
        for i in range(len(train_data)):
            # # 丢掉205.251.126.93看是否能消除nan

            # 	continue
            # 丢掉包含ip数量少的batch
            if train_data[i]["tg_Y"].shape[0] <= 6:
                continue

            count += 1

            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min, center = train_data[i]["lm_X"], \
                train_data[i]["lm_Y"], \
                train_data[i]["tg_X"], \
                train_data[i]["tg_Y"], \
                train_data[i]["lm_delay"], \
                train_data[i]["tg_delay"], \
                train_data[i]["y_max"], \
                train_data[i]["y_min"], \
                train_data[i]["center"]
            optimizer.zero_grad()

            # y_pred, adj, adj_teacher
            feature, pred, mask = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), Tensor(tg_Y), Tensor(lm_delay),
                                        Tensor(tg_delay))

            # if torch.isnan(pred).any():
            # 	print(f'last route{center}, have nan')
            # if torch.isinf(pred).any():
            # 	print(f'last route{center}, have inf')

            distance = dis_loss(Tensor(tg_Y), pred, y_max, y_min)

            loss_contrast = loss_fun(features=feature.unsqueeze(1), mask=mask)

            loss_mse = distance * distance  # mse loss
            loss_mse = loss_mse.sum()

            if torch.isnan(loss_contrast).any():
                print(f'loss_contrast：{loss_contrast}, have nan')
            if torch.isnan(loss_mse).any():
                print(f'loss_mse：{loss_mse}, have nan')

            # print(f"loss_contrast={loss_contrast}, loss_mse={loss_mse}")
            loss_all = opt.loss_contrast_ratio * loss_contrast + opt.loss_mse_ratio * loss_mse

            loss_all.requires_grad_(True)
            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # 获取梯度并打印
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f'Parameter: {name}, Gradient contains nan')

                        print(f'loss_contrast：{loss_contrast}, loss_mse：{loss_mse}')

                        print(train_data[i])

                        lab = 1  # 训练停止
                        break
                    if torch.isinf(param.grad).any():
                        print(f'Parameter: {name}, Gradient contains inf')
                        break
                # print(f'Parameter: {name}, Gradient: {param.grad}')
            if lab == 1:
                break
            optimizer.step()

            total_loss += loss_all.item()
            total_mae += loss_mse.sum()
            train_num += len(tg_Y)

        if lab == 1:
            break
        # 打印计算图
        # graph = make_dot(loss_all, params=dict(model.named_parameters()), show_attrs=True,
        #                  show_saved=True)
        # graph.render(filename='cnn', view=False, format='png')
        # break

        total_loss = total_loss / train_num
        total_mae = total_mae / train_num

        print("train: loss: {:.4f} mae: {:.4f}".format(total_loss, total_mae))

        # test
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []

        model.eval()
        distance_all = []
        with torch.no_grad():
            for i in range(len(test_data)):
                lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                    test_data[i][
                        "tg_X"], test_data[i]["tg_Y"], \
                    test_data[i][
                        "lm_delay"], test_data[i]["tg_delay"], \
                    test_data[i]["y_max"], test_data[i]["y_min"]
                # y_pred, adj, adj_teacher
                feature, pred, mask = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), Tensor(tg_Y), Tensor(lm_delay),
                                            Tensor(tg_delay))
                distance = dis_loss(Tensor(tg_Y), pred, y_max, y_min)
                for i in range(len(distance.cpu().detach().numpy())):
                    dislist.append(distance.cpu().detach().numpy()[i])
                    distance_all.append(distance.cpu().detach().numpy()[i])
                test_num += len(tg_Y)
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()

            total_mse = total_mse / test_num
            total_mae = total_mae / test_num

            print("test: mse: {:.4f}  mae: {:.4f}".format(total_mse, total_mae))
            dislist_sorted = sorted(dislist)
            print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])

            # save checkpoint
            if epoch > 0 and epoch % opt.saved_epoch == 0:
                savepath = f"{model_path}/{opt.dataset}_{epoch}.pth"
                save_cpt(model, optimizer, epoch, savepath)
                print("Save checkpoint!")

                f = open(
                    f"{log_path}/{opt.dataset}_{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-{current_time.minute}.txt",
                    'a')
                f.write(f"\n*********epoch={epoch}*********\n")
                f.write(
                    "test: mse: {:.3f}\trmse: {:.3f}\tmae: {:.3f}".format(total_mse, np.sqrt(total_mse.cpu().item()),
                                                                          total_mae))
                f.write("\ttest median: {:.3f}".format(dislist_sorted[int(len(dislist_sorted) / 2)]))

                current_time_2 = datetime.datetime.now()
                f.write(
                    f"\ttime:{current_time_2.year}-{current_time_2.month}-{current_time_2.day} {current_time_2.hour}:{current_time_2.minute}:{current_time_2.second}")
                f.close()

            batch_metric = total_mae.cpu().numpy()
            if batch_metric <= np.min(losses):

                '''
                Changes:加了回退机制
                '''
                # 保存最好的模型参数

                if num_back < 1:
                    second_model_state = model.state_dict()
                else:
                    second_model_state = copy.deepcopy(best_model_state)  # 回退到次优模型

                best_model_state = model.state_dict()  # 以便后续回到原点

                no_better_epoch = 0
                early_stop_epoch = 0
                print("Better MAE in epoch {}: {:.4f}".format(epoch, batch_metric))

                # 记录目前指标最好的模型：模型，指标
                savepath = f"{model_path}/{opt.dataset}_best.pth"
                save_cpt(model, optimizer, epoch, savepath)
                print("Save checkpoint!")

                f = open(
                    f"{log_path}/{opt.dataset}_{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-{current_time.minute}.txt",
                    'a')
                f.write(f"\n*********epoch={epoch}*********\n")
                f.write(
                    "test: mse: {:.3f}\trmse: {:.3f}\tmae: {:.3f}".format(total_mse, np.sqrt(total_mse.cpu().item()),
                                                                          total_mae))
                f.write("\ttest median: {:.3f}".format(dislist_sorted[int(len(dislist_sorted) / 2)]))

                current_time_2 = datetime.datetime.now()
                f.write(
                    f"\ttime:{current_time_2.year}-{current_time_2.month}-{current_time_2.day} {current_time_2.hour}:{current_time_2.minute}:{current_time_2.second}")
                f.close()

            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = early_stop_epoch + 1

            losses.append(batch_metric)

        # halve the learning rate
        if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
            if num_back < 1:
                lr = 2e-2
                print("learning rate changes to {}!\n".format(lr))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
                no_better_epoch = 0
                num_back += 1

            else:
                lr /= 2
                print("learning rate changes to {}!\n".format(lr))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
                no_better_epoch = 0
                num_back += 1

            '''
            Changes
            '''
            # 恢复到次优模型
            model.load_state_dict(second_model_state)
            no_better_epoch = 0
            print("我回退到了second best模型参数")

        if early_stop_epoch == opt.early_stop_epoch:
            break
