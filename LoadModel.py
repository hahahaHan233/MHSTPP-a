import random

import numpy as np
import pandas as pd
import torch
import logging
import os

from model import *
from config import *
from preprocess import *
from dataset import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device_str = "cpu"
if torch.cuda.is_available():
    device_str = "cuda:0"
elif torch.backends.mps.is_available():  # macos
    device_str = "mps:0"
device = torch.device(device_str)

create_log(log_dir, 'test')

for i in [1, 2]:
    print(f'+++++++++++++ Dataset: {dataset[i]} ++++++++++++++++')
    train_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_train.txt')
    test_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_test.txt')
    poi_position_path = os.path.join(save_path, dataset[i], dataset[i] + '_POI_position.txt')

    assert os.path.exists(train_set_path)
    assert os.path.exists(test_set_path)

    model = Model(dataset[i], train_set_path, test_set_path, poi_position_path,
                  emb_size=32,
                  neg_size=neg_size_list[i],
                  hist_len=hist_length_list[i],
                  user_count=user_cnt_list[i],
                  item_count=poi_cnt_list[i],
                  learning_rate=learning_rate_list[i],
                  decay=decay_list[i],
                  batch_size=batch_size_list[i],
                  epoch_num=epoch_num_list[i],
                  top_n=20,
                  num_workers=0,
                  device=device)

    # load model
    source_dir = './model/'
    log_str = '2024-01-08_21-27-42_FULL_emb32'
    # log_str = '2024-01-08_03-54-32_FULL'

    model.load_state_dict(torch.load(
        os.path.join(source_dir, f'{log_str}_{dataset[i]}.pth')))

    # [recall,mrr] = model.evaluate(-1)

    embeddings = model.emb.weight.data.cpu().numpy()

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    csv_file = f'./processed_data/{dataset[i]}/{dataset[i]}_test.txt'
    col_names = ['uid', 'poi', 'timestamp', 'lat', 'lon']
    dtypes = {'uid': np.int64, 'poi': np.int64, 'timestamp': np.float64, 'lat': np.float64, 'lon': np.float64}
    df = pd.read_csv(csv_file, sep='\t', header=None, names=col_names, encoding='utf8').astype(dtypes)

    # 设置随机选取的用户数量
    n = 10  # 或者任何你需要的用户数量

    # 确保不要选取超过实际用户数的数量
    actual_users_count = df['uid'].nunique()
    n = min(n, actual_users_count)

    # 从所有用户中随机选择n个用户
    selected_users = random.sample(list(df['uid'].unique()), n)

    # 为每个选定的用户分配颜色和标记
    # 定义一个标记列表
    markers = ['o', 's', '^', 'p', '*', 'D', 'v', 'x', '+', '<', '>']
    # 如果选定的用户数量大于标记列表的长度，可能需要重复标记列表或选择更多标记
    colors = plt.cm.rainbow(np.linspace(0, 1, n))

    # 筛选出这些用户的数据
    selected_data = df[df['uid'].isin(selected_users)]

    import itertools

    # 绘制选定用户的经纬度坐标
    plt.figure(figsize=(10, 6))
    for user, color, marker in zip(selected_users, colors, itertools.cycle(markers)):
        user_data = selected_data[selected_data['uid'] == user]
        plt.scatter(user_data['lon'], user_data['lat'], color=color, marker=marker, label=f'User {user}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Visualization of Selected User Visits on Geographic Coordinates')
    plt.legend()
    plt.show()

    user_item_mapping = df.groupby('uid')['poi'].apply(list).to_dict()

    # 步骤2: 随机采样n个用户和m个项目
    markers = ['o', 's', '^', 'p', '*', '+', 'x', 'D', 'h', '>', '<']
    n = 8  # 随机选择的用户数
    m = 100  # 每个用户选择的项目数
    random.seed(10)
    selected_users = random.sample(user_item_mapping.keys(), n)
    #selected_users = [5668, 5670, 5669, 5672, 5673, 5675]
    sampled_user_item = {user: random.sample(user_item_mapping[user], min(len(user_item_mapping[user]), m)) for user in
                         selected_users}

    user = []

    # 步骤3: 处理重复项目
    unique_items = set()
    for user, items in sampled_user_item.items():
        new_items = [item for item in items if item not in unique_items]
        unique_items.update(new_items)
        sampled_user_item[user] = new_items

    all_sampled_items = set(item for items in sampled_user_item.values() for item in items)
    sampled_embeddings = embeddings[list(all_sampled_items)]

    # 使用t-SNE降维到二维
    tsne = TSNE(n_components=2, perplexity=4, n_iter=300, random_state=33, learning_rate='auto')
    reduced_embeddings = tsne.fit_transform(sampled_embeddings)

    # 创建一个项目ID到t-SNE嵌入的映射
    item_to_tsne = dict(zip(all_sampled_items, reduced_embeddings))

    # 可视化
    plt.figure(figsize=(10, 8))
    for (user, items), marker in zip(sampled_user_item.items(), markers):
        item_embeddings = np.array([item_to_tsne[item] for item in items if item in item_to_tsne])
        plt.scatter(item_embeddings[:, 0], item_embeddings[:, 1], marker=marker, label=f'User {user}')

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of User Interactions')
    plt.legend()
    plt.show()

    break
