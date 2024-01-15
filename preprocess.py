import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
import time
import hashlib
from config import *

HASH_TYPE = 'sha256'


def generate_file_hash(file_path):
    """
    Generate the hash value for the local file.
    """
    hash_obj = hashlib.new(HASH_TYPE)
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def process(file_path, save_path, user_thr=10, poi_thr=10, seq_len=100, norm_method='hour', hist_len=None):
    """
    Process the input data. Split train-test set by 8-2.

    Parameters:
    - file_path (str): The file path representing the input data.
    - save_path (str): Save path for the processed data.
    - user_thr (float, optional): User under this frequency should be eliminated (default: user_thr).
    - poi_thr (float, optional): Point of interest under this frequency should be eliminated (default: poi_thr).
    - seq_len (int, optional): Sequence length for the latest checkIn history (default: seq_len).
    - norm_method (str, optional): Normalization for the time interval (default: norm_method).
    - hist_len (str, None): Context window length for the target in the test set (default: None).

    Returns:
    None
    """

    # print the specified parameters
    print(f"file_path: {file_path}")
    print(f"save_path: {save_path}")
    print(f"user_thr: {user_thr}")
    print(f"poi_thr: {poi_thr}")
    print(f"seq_len: {seq_len}")
    print(f"norm_method: {norm_method}")
    print(f"hist_len: {hist_len}")

    col_names = ['uid', 'poi', 'cat_id', 'category', 'lat', 'lon', 'offset', 'time', 'unixTime', 'dayOff', 'cid']
    df = pd.read_csv(file_path, sep='\t', header=None, names=col_names, encoding='utf8')
    df = df[['uid', 'poi', 'lat', 'lon', 'unixTime']]

    # remove the user with the insufficient frequency
    user_counts = df['uid'].value_counts()
    valid_users = user_counts[user_counts >= user_thr].index
    df = df[df['uid'].isin(valid_users)]

    # remove the POI with the insufficient frequency
    poi_counts = df['poi'].value_counts()
    valid_poi = poi_counts[poi_counts >= poi_thr].index
    df = df[df['poi'].isin(valid_poi)]

    # keep the fixed length of history as seq_len
    group_list = []
    for uid, group in df.groupby(by='uid'):
        group = group.reset_index(drop=True, inplace=False).copy()
        group.sort_values(by=['unixTime'], ascending=True, inplace=True)

        # Keep the latest history with length <= seq_len
        if len(group) > seq_len:
            group = group[len(group) - seq_len:]
        assert len(group) <= seq_len
        group_list.append(group.copy())

    df = pd.concat(group_list, axis=0)
    df.reset_index(drop=True, inplace=True)

    # reindex the uid and POI (Some users and POIs are removed)
    df['uid'] = df['uid'].astype('category').cat.codes
    df['poi'] = df['poi'].astype('category').cat.codes
    # reindex start from 0!!
    assert len(df['uid'].unique()) == df['uid'].max() + 1
    assert len(df['poi'].unique()) == df['poi'].max() + 1
    print('Number of user:{}'.format(len(df['uid'].unique())))
    print('Number of item:{}'.format(len(df['poi'].unique())))
    print('Number of checkIns:{}'.format(len(df)))
    # user and POI share the same index space
    # df['poi'] = df['poi'] + len(df['uid'].unique())
    # assert len(df['uid'].unique()) + len(df['poi'].unique()) == df['poi'].max() + 1
    df['uid'] = df['uid'] + len(df['poi'].unique())
    assert len(df['uid'].unique()) + len(df['poi'].unique()) == df['uid'].max() + 1

    # calculate the difference between timestamp
    df.sort_values(['uid', 'unixTime'], inplace=True)
    df['time_diff'] = df.groupby('uid')['unixTime'].diff().fillna(0)  # The begining is set to 0

    # normalization the time interval:log - hour - mm
    assert norm_method in ['log', 'mm', 'hour'], 'Not defined normalization method!'
    if norm_method == 'log':
        df['time_diff_norm'] = np.log(df['time_diff'] + 1.0 + 1e-6)  # log limit + 1.0 + 1e-6
    elif norm_method == 'mm':
        # map the interval to [0,1] according to user
        min_values = df.groupby('uid')['time_diff'].min()
        max_values = df.groupby('uid')['time_diff'].max()
        df['time_diff_norm'] = df.apply(
            lambda row: (row['time_diff'] - min_values[row['uid']]) / (max_values[row['uid']] - min_values[row['uid']]),
            axis=1)
    elif norm_method == 'hour':
        df['time_diff_norm'] = df['time_diff'] / 3600

    # num_list = np.zeros(10)
    # for d_time in df['time_diff']:
    #     for i in range(10):
    #         if d_time >= 10 ** (i-1) and d_time <= 10 ** i:
    #             num_list[i]+=1
    # print(num_list)


    # accumulate the difference of time to get new timestamp
    df['cumulative_time_diff_norm'] = df.groupby('uid')['time_diff_norm'].cumsum()

    # split train-test set (8-2)
    train_list, test_list = [], []
    for user_id in df['uid'].unique():
        user_data = df[df['uid'] == user_id]
        split_index = int(len(user_data) * 0.8)

        train_data = user_data.iloc[:split_index]
        # get the context window from training sequence, preserve target in the test sequence
        if hist_len != None:
            test_data = user_data.iloc[max(0, split_index - hist_len):]
        else:
            test_data = user_data.iloc[split_index:]

        # Each user test length no more than this limit
        assert len(test_data) <= int(seq_len * 0.2) + hist_len

        train_list.append(train_data)
        test_list.append(test_data)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    # check the length of extended test data with hist
    # print((len(hist_test_df) - len(test_df) )/ len(df['uid'].unique()) )

    '''save to local'''
    # save_path/
    # ├── dataset1/
    # │   ├── dataset1_train.txt
    # │   ├── dataset1_test.txt
    # │   ├── dataset1_POI_postion.txt
    # │   └── dataset1_info.txt
    # ├── dataset2/
    # │   ├── dataset2_train.txt
    # │   ├── dataset2_test.txt
    # │   ├── dataset2_POI_postion.txt
    # │   └── dataset2_info.txt
    # └── ...

    # create dataInfo file to store the dataset details
    dataset_name = os.path.basename(file_path).split('_')[0]
    dir_path = os.path.join(save_path, dataset_name)
    if os.path.exists(dir_path): shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    # store the all POI position
    all_POI_pos = df[['poi', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)
    all_POI_pos = all_POI_pos.groupby('poi').mean().reset_index()
    assert len(all_POI_pos) == df['poi'].max() + 1
    with open(os.path.join(dir_path, dataset_name + '_POI_position.txt'), 'w', newline="", encoding='utf8') as csv_file:
        all_POI_pos.to_csv(path_or_buf=csv_file, index=False, header=False, sep='\t')

    # store the train and test sets
    col_names_save = ['uid', 'poi', 'cumulative_time_diff_norm', 'lat', 'lon']
    with open(os.path.join(dir_path, dataset_name + '_train.txt'), 'w', newline="", encoding='utf8') as csv_file:
        train_df[col_names_save].to_csv(path_or_buf=csv_file, index=False, header=False, sep='\t')
    with open(os.path.join(dir_path, dataset_name + '_test.txt'), 'w', newline="", encoding='utf8') as csv_file:
        test_df[col_names_save].to_csv(path_or_buf=csv_file, index=False, header=False, sep='\t')

    # create data info
    current_timestamp = time.time()
    current_datetime = datetime.fromtimestamp(current_timestamp)
    dataset_info = f"{current_datetime} \n"
    dataset_info += f"Dataset Information: {dataset_name} \n"
    dataset_info += f"Dataset Path: {file_path}\n"
    dataset_info += f"user_thr={user_thr}, poi_thr={poi_thr}, seq_len={seq_len}, norm_method={norm_method}, hist_len={hist_len}\n"
    dataset_info += f"Number of Users: {len(df['uid'].unique())}\n"
    dataset_info += f"Number of POIs: {len(df['poi'].unique())}\n"
    dataset_info += f"Number of CheckIns: {len(df)}\n"
    dataset_info += f"Number of train set ChekIns: {len(train_df)}\n"
    dataset_info += f"Number of test set ChekIns: {len(test_df)}\n"
    dataset_info += f"Dataset Hash: {generate_file_hash(file_path)}\n"
    dataset_info += f"Train dataset Hash: {generate_file_hash(os.path.join(dir_path, dataset_name + '_train.txt'))}\n"
    dataset_info += f"Test dataset Hash: {generate_file_hash(os.path.join(dir_path, dataset_name + '_test.txt'))}\n"

    with open(os.path.join(dir_path, dataset_name + '_datasetInfo.txt'), 'w') as file:
        file.write(dataset_info)

    print(f"Dataset information saved to {dataset_name + '_datasetInfo.txt'}.")


if __name__ == '__main__':
    file_path_list = ['./data/NYC_with_dayOff.txt', './data/TKY_with_573703_dayOff.txt',
                      './data/SIN_with_356381_dayOff.txt']
    save_path = './processed_data'

    user_thr = 10
    poi_thr = 10
    seq_len = 100
    hist_len = 32
    norm_method = 'hour'

    for i, file_path in enumerate(file_path_list):
        print('-----------------------------------------')
        print('1.Start preprocess data...')
        process(file_path_list[i], save_path, user_thr=user_thr_list[i], poi_thr=poi_thr_list[i],
                seq_len=seq_len_list[i], norm_method=norm_method_list[i],
                hist_len=hist_length_list[i])
        print('2.Finish processed data successfully!')
        # print(generate_file_hash(file_path))
