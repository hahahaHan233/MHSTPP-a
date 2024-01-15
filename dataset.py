import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
import random


class DataSetTrain(Dataset):

    def __init__(self, train_path, user_count=0, item_count=0, neg_size=5, hist_len=2):

        """parameters"""
        self.neg_size = neg_size
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len

        """data"""
        logging.info('Start generating training sample...')
        self.train_path = train_path
        self.NEG_SAMPLING_POWER = 0.75
        self.user2negSamples = {}
        self.user2negDistribution = {}
        self.init_negative()

        # batch
        self.source_nodes = []
        self.target_nodes = []
        self.target_lat_pos = []
        self.target_lon_pos = []
        self.target_times = []

        # batch x hist
        self.hist_nodes = []
        self.hist_masks = []
        self.hist_times = []
        self.hist_lat_pos = []
        self.hist_lon_pos = []

        # batch x neg_sample
        self.neg_nodes = []

        # generate sample for training
        self.init_sample()

        assert len(self.source_nodes) == len(self.target_nodes) == len(self.hist_nodes) == len(self.hist_lon_pos)
        self.sample_size = len(self.source_nodes)

        logging.info('Finish generating training sample...')

    def init_sample(self):
        """ Generate samples """
        col_names = ['uid', 'poi', 'timestamp', 'lat', 'lon']
        dtypes = {'uid': np.int64, 'poi': np.int64, 'timestamp': np.float64, 'lat': np.float64, 'lon': np.float64}
        df = pd.read_csv(self.train_path, sep='\t', header=None, names=col_names, encoding='utf8')

        for uid, group in df.groupby(by='uid'):
            user_sequence = group[['uid', 'poi', 'timestamp', 'lat', 'lon']].astype(dtypes).values

            for i in range(len(user_sequence)):
                hist_masks = np.zeros((self.hist_len)).astype(int)
                hist_nodes = np.zeros((self.hist_len,)).astype(int)
                hist_times = np.zeros((self.hist_len,))
                hist_lat_pos = np.zeros((self.hist_len,))
                hist_lon_pos = np.zeros((self.hist_len,))

                if i < self.hist_len:
                    # need padding
                    hist_masks[self.hist_len - i:] = 1
                    hist_nodes[self.hist_len - i:] = user_sequence[:i, 1]
                    hist_times[self.hist_len - i:] = user_sequence[:i, 2]
                    hist_lat_pos[self.hist_len - i:] = user_sequence[:i, 3]
                    hist_lon_pos[self.hist_len - i:] = user_sequence[:i, 4]
                else:
                    hist_masks[:] = 1
                    hist_nodes[:] = user_sequence[i - self.hist_len:i, 1]
                    hist_times[:] = user_sequence[i - self.hist_len:i, 2]
                    hist_lat_pos[:] = user_sequence[i - self.hist_len:i, 3]
                    hist_lon_pos[:] = user_sequence[i - self.hist_len:i, 4]

                self.source_nodes.append(uid)
                self.target_nodes.append(user_sequence[i, 1])
                self.target_times.append(user_sequence[i, 2])
                self.target_lat_pos.append(user_sequence[i, 3])
                self.target_lon_pos.append(user_sequence[i, 4])

                self.hist_nodes.append(hist_nodes)
                self.hist_masks.append(hist_masks)
                self.hist_times.append(hist_times)
                self.hist_lat_pos.append(hist_lat_pos)
                self.hist_lon_pos.append(hist_lon_pos)

                neg_nodes = np.random.choice(self.user2negSamples[uid], size=self.neg_size,
                                             p=self.user2negDistribution[uid], replace=False)

                # for i in neg_nodes:
                #     assert i not in group['poi'].unique()

                self.neg_nodes.append(neg_nodes)

    def init_negative(self):
        # Generate negative samples according to each user
        # Rule: The highly frequent item that is not interacted by user, \
        # is regarded as negative sample to this user / disliked by this user.

        col_names = ['uid', 'poi', 'timestamp', 'lat', 'lon']
        df = pd.read_csv(self.train_path, sep='\t', header=None, names=col_names, encoding='utf8')

        # get the frequency of POI
        item_counts = df['poi'].value_counts()

        all_items = set(df['poi'].unique())
        for uid, group in df.groupby(by='uid'):
            user_interacted_items = set(group['poi'].unique())
            not_interacted_items = all_items - user_interacted_items

            item_freqs = item_counts[list(not_interacted_items)].sort_values(ascending=False)
            item_freqs = item_freqs ** self.NEG_SAMPLING_POWER
            item_probs = item_freqs / item_freqs.sum()

            self.user2negSamples[uid] = item_probs.index  # POI id
            self.user2negDistribution[uid] = item_probs.values  # POI prob

        return

    def __len__(self):
        '''number of samples'''
        return self.sample_size

    def __getitem__(self, idx):
        ''' get one item'''

        sample = {
            'source_node': self.source_nodes[idx],
            'target_node': self.target_nodes[idx],
            'target_time': self.target_times[idx],
            'target_loc_lat': self.target_lat_pos[idx],
            'target_loc_lon': self.target_lon_pos[idx],
            'history_nodes': self.hist_nodes[idx],
            'history_times': self.hist_times[idx],
            'history_locs_lat': self.hist_lat_pos[idx],
            'history_locs_lon': self.hist_lon_pos[idx],
            'history_masks': self.hist_masks[idx],
            'neg_nodes': self.neg_nodes[idx],
        }

        return sample


class DataSetTest(Dataset):
    def __init__(self, test_path, user_count=0, item_count=0, hist_len=2):

        """parameters"""
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len

        """data"""
        logging.info('Start generating testing sample...')
        self.test_path = test_path

        # batch
        self.source_nodes = []
        self.target_nodes = []
        self.target_lat_pos = []
        self.target_lon_pos = []
        self.target_times = []

        # batch x hist
        self.hist_nodes = []
        self.hist_masks = []
        self.hist_times = []
        self.hist_lat_pos = []
        self.hist_lon_pos = []

        # generate sample for training
        self.init_sample()

        assert len(self.source_nodes) == len(self.target_nodes) == len(self.hist_nodes) == len(self.hist_lon_pos)
        self.sample_size = len(self.source_nodes)

        logging.info('Finish generating testing sample...')

    def init_sample(self):
        """ Generate samples """
        col_names = ['uid', 'poi', 'timestamp', 'lat', 'lon']
        dtypes = {'uid': np.int64, 'poi': np.int64, 'timestamp': np.float64, 'lat': np.float64, 'lon': np.float64}
        df = pd.read_csv(self.test_path, sep='\t', header=None, names=col_names, encoding='utf8')

        for uid, group in df.groupby(by='uid'):
            user_sequence = group[['uid', 'poi', 'timestamp', 'lat', 'lon']].astype(dtypes).values

            for i in range(self.hist_len, len(user_sequence)): # start from context window from training set
                hist_masks = np.zeros((self.hist_len)).astype(int)
                hist_nodes = np.zeros((self.hist_len,)).astype(int)
                hist_times = np.zeros((self.hist_len,))
                hist_lat_pos = np.zeros((self.hist_len,))
                hist_lon_pos = np.zeros((self.hist_len,))

                if i < self.hist_len:
                    # need padding
                    hist_masks[self.hist_len - i:] = 1
                    hist_nodes[self.hist_len - i:] = user_sequence[:i, 1]
                    hist_times[self.hist_len - i:] = user_sequence[:i, 2]
                    hist_lat_pos[self.hist_len - i:] = user_sequence[:i, 3]
                    hist_lon_pos[self.hist_len - i:] = user_sequence[:i, 4]
                else:
                    hist_masks[:] = 1
                    hist_nodes[:] = user_sequence[i - self.hist_len:i, 1]
                    hist_times[:] = user_sequence[i - self.hist_len:i, 2]
                    hist_lat_pos[:] = user_sequence[i - self.hist_len:i, 3]
                    hist_lon_pos[:] = user_sequence[i - self.hist_len:i, 4]

                self.source_nodes.append(uid)
                self.target_nodes.append(user_sequence[i, 1])
                self.target_times.append(user_sequence[i, 2])
                self.target_lat_pos.append(user_sequence[i, 3])
                self.target_lon_pos.append(user_sequence[i, 4])

                self.hist_nodes.append(hist_nodes)
                self.hist_masks.append(hist_masks)
                self.hist_times.append(hist_times)
                self.hist_lat_pos.append(hist_lat_pos)
                self.hist_lon_pos.append(hist_lon_pos)

    def __len__(self):
        '''number of samples'''
        return self.sample_size

    def __getitem__(self, idx):

        sample = {
            'source_node': self.source_nodes[idx],
            'target_node': self.target_nodes[idx],
            'target_time': self.target_times[idx],
            'target_loc_lat': self.target_lat_pos[idx],
            'target_loc_lon': self.target_lon_pos[idx],
            'history_nodes': self.hist_nodes[idx],
            'history_times': self.hist_times[idx],
            'history_locs_lat': self.hist_lat_pos[idx],
            'history_locs_lon': self.hist_lon_pos[idx],
            'history_masks': self.hist_masks[idx],
        }

        return sample
