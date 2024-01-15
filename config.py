# -*- coding: utf-8 -*-
from datetime import datetime

FORMAT = "%(asctime)s - %(message)s"
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = './results'
log_str = current_time + '_' + f'FULL'

# neg_num = 4
# neg_size_list = [neg_num, neg_num, neg_num]
# log_str = current_time + '_' + f'FULL_Neg{neg_num}'

# emb_dim = 32
# emb_size_list = [emb_dim, emb_dim, emb_dim]
# log_str = current_time + '_' + f'FULL_emb{emb_dim}'


generate_dataset = False
generate_sample = False
mode = 'Train'  # or 'Test'

device_list = ['cuda:0', 'mps:0']
dataset = ['NYC', 'TKY', 'SIN']

# Data process
dataset_hash = ['6f6581f741adace1724e91e675c1ddf9fe4d60c6083d02d8ccf512687be756da',
                '9060bd6ab75eb44e7f88aa8941e6be722cf2b74a459b5ecd8727d4fcf5109a92',
                'aac7ecc9759b5c2fe9472c716651e9fe1a46b40d231a01118e32a1f03059d3df']

file_path_list = ['./data/NYC_with_dayOff.txt', './data/TKY_with_573703_dayOff.txt',
                  './data/SIN_with_356381_dayOff.txt']
save_path = './processed_data'

topN = [1, 5, 10, 20]

layout = {
    "Loss": {
        "Training Loss": ["Multiline", ["loss/train"]],
        "Validation Loss": ["Multiline", ["loss/validate"]],
    },
    "Metrics": {
        "Recall": ["Multiline", [f"Recall/top@{N}" for N in topN]],
        "MRR": ["Multiline", [f"MRR/top@{N}" for N in topN]],
    },
}

user_thr_list = [10, 10, 10]
poi_thr_list = [10, 10, 10]
user_cnt_list = [1083, 2293, 3745]
poi_cnt_list = [5002, 7676, 5511]
seq_len_list = [100, 100, 100]

norm_method_list = ['hour', 'hour', 'hour']
# norm_method_list = ['mm', 'mm', 'mm']
hist_length_list = [32, 32, 32]  # no more than seq_lenth

# Model
emb_size_list = [512, 512, 512]
neg_size_list = [5, 5, 5]

learning_rate_list = [0.0005, 0.0005, 0.0005]
decay_list = [0.01, 0.01, 0.01]

# training
batch_size_list = [512, 512, 512]
epoch_num_list = [50, 50, 50]

