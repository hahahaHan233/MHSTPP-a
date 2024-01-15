import logging
import os
import torch
import math
import pandas as pd
import numpy as np

FORMAT = "%(asctime)s - %(message)s"
def create_log(log_dir, log_name):
    """Create log output to console and file"""
    if os.path.exists(log_dir) == False: os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_name + '.txt')

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter(FORMAT)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])


def metrix2str(recall, MRR):
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    metric = np.array([1, 5, 10, 20])

    header = "top@N\tRecall\tMRR"
    output = f"{header}\n"
    for value, recall, mrr in zip(metric, recall[metric - 1], MRR[metric - 1]):
        output += f"{value}\t{recall:.4f}\t{mrr:.4f}\n"

    return output

    # np_obj = np_obj.reshape(-1,5)
    # for row in np_obj:
    #     if log:
    #         logging.info(row[metric])
    #     else:
    #         print(row)


def haversine(lon_1, lat_1, lon_2, lat_2):
    # 定义 radians 函数
    def radians(degrees):
        return degrees * (math.pi / 180)

    # 使用 torch.Tensor.apply_() 将 radians 函数应用到每个元素
    lon_1.cpu().apply_(radians)
    lat_1.cpu().apply_(radians)
    lon_2.cpu().apply_(radians)
    lat_2.cpu().apply_(radians)

    dlon = torch.abs(lon_2 - lon_1)
    dlat = torch.abs(lat_2 - lat_1)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_1) * torch.cos(lat_2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    r = 6371

    return c * r


def generate_averagePos(file_path):
    col_names = ['uid', 'poi', 'timestamp', 'lat', 'lon']
    df = pd.read_csv(file_path, sep='\t', header=None, names=col_names, encoding='utf8')

    pos = df[['poi', 'lat', 'lon']].drop_duplicates(subset=['poi']).reset_index()
    global avg_lat, avg_lon
    avg_lat, avg_lon = pos[['lat', 'lon']].mean()

    return avg_lat, avg_lon


def get_all_position(poi_position_path):
    '''return lat,lon tensor'''
    file_path = poi_position_path
    col_names = ['poi', 'lat', 'lon']
    df = pd.read_csv(file_path, sep='\t', header=None, names=col_names, encoding='utf8')
    df = df.sort_values(by='poi').reset_index(drop=True)

    lat_tensor = torch.tensor(df['lat'].to_numpy(), dtype=torch.float32)
    lon_tensor = torch.tensor(df['lon'].to_numpy(), dtype=torch.float32)

    all_pos_lat = torch.deg2rad(lat_tensor)  # item_num
    all_pos_lon = torch.deg2rad(lon_tensor)  # item_num

    # item_num x item_num
    d_pos = 2 * 6371 * torch.arcsin(torch.sqrt( \
        torch.sin((all_pos_lat - all_pos_lat.unsqueeze(dim=1)) / 2) ** 2 + \
        torch.cos(all_pos_lat) * torch.cos(all_pos_lat) * \
        torch.sin((all_pos_lon - all_pos_lon.unsqueeze(dim=1)) / 2) ** 2))

    # return lat_tensor,lon_tensor
    return d_pos


def cal_haversineDist(lat_1, lon_1, lat_2, lon_2):
    lat_1 = torch.deg2rad(lat_1)
    lon_1 = torch.deg2rad(lon_1)
    lat_2 = torch.deg2rad(lat_2)
    lon_2 = torch.deg2rad(lon_2)

    d_pos = 2 * 6371 * torch.arcsin(torch.sqrt( \
        torch.sin((lat_1 - lat_2) / 2) ** 2 + \
        torch.cos(lat_1) * torch.cos(lat_2) * \
        torch.sin((lon_1 - lon_2) / 2) ** 2))

    return d_pos


def min_max_normalize(distance_matrix):
    min_value = torch.min(distance_matrix)
    max_value = torch.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_value) / (max_value - min_value)
    return normalized_matrix


def z_score_normalize(distance_matrix):
    mean_value = torch.mean(distance_matrix)
    std_value = torch.std(distance_matrix)
    normalized_matrix = (distance_matrix - mean_value) / std_value
    return normalized_matrix


def softmax_normalize(distance_matrix):
    normalized_matrix = torch.nn.functional.softmax(distance_matrix, dim=1)
    return normalized_matrix


def print_model_size(model):
    total_bytes = 0
    print('------------------------------------------')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_size = parameter.numel()
            param_bytes = param_size * parameter.element_size()
            total_bytes += param_bytes
            print(f"{name}: {param_bytes} B")
    print('------------------------------------------')
    print(f"Total trainable parameters: {total_bytes / (1024 * 1024):.4f} MB ")


class val_metric:  # 评价矩阵类
    def __init__(self, top_K):
        self.top_K = [1, 5, 10, 15, 20]
        self.results = {}
        for K in self.top_K:
            self.results['epoch%d' % K] = [0, 0]
            self.results['metric%d' % K] = [0, 0]
        for K in self.top_K:
            self.results['hit%d' % K] = []
            self.results['mrr%d' % K] = []

    def add_result(self, tar, scores):  # 更新字典
        index = torch.argsort(scores, dim=1, descending=True)
        index = index[:, :max(self.top_K)]  # 截取
        index = index.cpu().detach().numpy()
        # scores = scores.cpu().detach().numpy()

        tar = tar.cpu().detach().numpy()
        # index = []
        # for idd in range(scores.shape[0]):
        #     index.append(find_k_largest(30, scores[idd]))

        # index = np.argsort(-scores, 1)  # [:, -max(top_k):]
        # index = index[:, :max(self.top_K)]  # 截取

        for K in self.top_K:
            for prediction, target in zip(index[:, :K], tar):
                self.results['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    self.results['mrr%d' % K].append(0)
                else:
                    self.results['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))

    def show_metric(self, metrics, total_loss, epoch):  # 更新最佳结果
        for K in self.top_K:
            metrics.results['hit%d' % K] = np.mean(metrics.results['hit%d' % K]) * 100
            metrics.results['mrr%d' % K] = np.mean(metrics.results['mrr%d' % K]) * 100

        if self.results['metric%d' % self.top_K[-1]][0] <= metrics.results['hit%d' % self.top_K[-1]]:  # 按recall更新
            self.results['epoch'] = epoch
            self.results['loss'] = total_loss
            for K in self.top_K:
                self.results['metric%d' % K][0] = metrics.results['hit%d' % K]
                self.results['metric%d' % K][1] = metrics.results['mrr%d' % K]
        # print(metrics)
        print(f'epoch:{epoch}')
        for K in self.top_K:
            # print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
            #       (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
            #        best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f' %
                  (total_loss, K, metrics.results['hit%d' % K], K, metrics.results['mrr%d' % K]))  # 当前
        print('best recall in epoch:%d' % (self.results['epoch']))
        copy_recall, copy_mrr = [], []
        for K in self.top_K:
            # print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
            #       (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
            #        best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f' %
                  (self.results['loss'], K, self.results['metric%d' % K][0], K, self.results['metric%d' % K][1]))
            copy_recall.append(self.results['metric%d' % K][0])
            copy_mrr.append(self.results['metric%d' % K][1])
        for rate in copy_recall + copy_mrr:
            print(f'{rate:.4f}\t', end='')
        print('')
