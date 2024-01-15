from torch.utils.data import DataLoader
import torch
from config import *
from utils import *
import torch
import os

FType = torch.FloatTensor
LType = torch.LongTensor

top_n = 20

count_all = 0
recall_all = np.zeros(top_n, dtype=float)
MRR_all = np.zeros(top_n, dtype=float)

for i, dataset_name in enumerate(dataset):
    print(f'+++++++++++++ Dataset: {dataset_name} ++++++++++++++++')

    poi_position_path = f'D:\Project\Hawkes\Hawkess-POI\processed_data\{dataset_name}\{dataset_name}_POI_position.txt'
    # poi_position_path = os.path.join(save_path, dataset[i], dataset[i] + '_POI_position.txt')
    all_dist = get_all_position(poi_position_path)  # item_num x item_num

    # print(all_dist)
    # print(all_dist.mean(),all_dist.std())

    # num_list = np.zeros(10)
    # r,c = all_dist.shape
    # for i in range(r):
    #     for j in range(c):
    #         for k in range(10):
    #             if all_dist[i,j] >= 10 ** k and all_dist[i,j] <= 10 ** (k + 1):
    #                 num_list[k]+=1
    # print(num_list)

    data_te = torch.load(f'../processed_data/{dataset_name}_te.pth')
    loader = DataLoader(data_te, batch_size=batch_size_list[i], shuffle=False, num_workers=0)

    for i_batch, sample_batched in enumerate(loader):
        # traverse history nodes find their nearest nodes
        hist_nodes = sample_batched['history_nodes'].type(LType)  # batch_size x hist_length
        dist_hist = all_dist[hist_nodes, :]  # batch_size x hist_length x item_num

        #dist_hist = all_dist[hist_nodes.unsqueeze(dim=1), hist_nodes.unsqueeze(dim=-1)] # batch_size x hist_length x hist_length

        #p_lambdas = dist_hist.mean(dim=1)
        p_lambdas = dist_hist[:,-1,:]

        sorted_poi_scores = torch.argsort(p_lambdas, dim=1, descending=False)  # batch_size x item_num

        # batch = 40
        # print(p_lambdas[batch,sorted_poi_scores[batch,0]],p_lambdas[batch,sorted_poi_scores[batch,6]])

        _, col_index = torch.where(sorted_poi_scores == \
                                   sample_batched['target_node'].type(LType).unsqueeze(
                                       dim=-1))  # batch_size

        col_index = col_index[col_index <= top_n - 1]

        col_index = col_index.cpu().numpy().tolist()
        recall = np.zeros(top_n, dtype=float)
        MRR = np.zeros(top_n, dtype=float)

        for target_index in col_index:
            recall[target_index:] += 1
            MRR[target_index:] += 1. / (target_index + 1)

        recall_all += recall
        MRR_all += MRR
        count_all += len(sample_batched['target_node'])

    recall_all = recall_all * 1. / count_all
    MRR_all = MRR_all * 1. / count_all

    print(f'{metrix2str(recall_all, MRR_all)}')
