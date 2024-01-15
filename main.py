# -*- coding: utf-8 -*-
import logging
import os

from config import *
import numpy as np
import torch
from preprocess import *
from dataset import *

from utils import *
from torch.utils.tensorboard import SummaryWriter

# loaded_modules = sys.modules.keys()
# print("imported modules:")
# for module in loaded_modules:
#     print(module)
# exit(0)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device_str = "cpu"
if torch.cuda.is_available():
    device_str = "cuda:0"
elif torch.backends.mps.is_available():  # macos
    device_str = "mps:0"
device = torch.device(device_str)

log_dir = './logs'
create_log(log_dir, log_str)

if __name__ == '__main__':
    from model import *

    logging.info(f'comment:{log_str}')
    logging.info(f'device:{device_str}')

    recall_list, mrr_list = [], []

    for i in [0, 1, 2]:
        logging.info(f'+++++++++++++ Dataset: {dataset[i]} ++++++++++++++++')

        log_path = f'./logs/{log_str}_{dataset[i]}'
        writer = SummaryWriter(log_dir=f'./logs/{log_str}_{dataset[i]}')
        writer.add_custom_scalars(layout)

        # check the dataset and local files
        assert generate_file_hash(file_path_list[i]) == \
               dataset_hash[i], 'The source data mismatch!'  # check source data
        dataset_dir = os.path.join(save_path, dataset[i])

        dataInfo_file_path = os.path.join(save_path, dataset[i], dataset[i] + '_datasetInfo.txt')
        if generate_dataset == True:
            # generate from source data
            logging.info('1.Start preprocess data...')
            process(file_path_list[i], save_path, user_thr=user_thr_list[i], poi_thr=poi_thr_list[i],
                    seq_len=seq_len_list[i], norm_method=norm_method_list[i],
                    hist_len=hist_length_list[i])
            logging.info('2.Finish processed data successfully!')
        elif os.path.exists(dataInfo_file_path):
            # load the local data
            logging.info('1.Check the local data...')
            with open(dataInfo_file_path, 'r') as infile:
                lines = infile.readlines()
                assert dataset_hash[i] == lines[-3].split('Dataset Hash:')[1].split()[0]
                assert lines[-2].split('Train dataset Hash:')[1].split()[0] \
                       == generate_file_hash(os.path.join(dataset_dir, dataset[i] + '_train.txt'))
                assert lines[-1].split('Test dataset Hash:')[1].split()[0] \
                       == generate_file_hash(os.path.join(dataset_dir, dataset[i] + '_test.txt'))
            logging.info('2.Check preprocessed data successfully!')

        # run model
        logging.info('3.Start running model...')

        train_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_train.txt')
        test_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_test.txt')
        poi_position_path = os.path.join(save_path, dataset[i], dataset[i] + '_POI_position.txt')

        assert os.path.exists(train_set_path)
        assert os.path.exists(test_set_path)

        train_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_train.txt')
        test_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_test.txt')
        poi_position_path = os.path.join(save_path, dataset[i], dataset[i] + '_POI_position.txt')

        assert os.path.exists(train_set_path)
        assert os.path.exists(test_set_path)

        model = Model(dataset[i], train_set_path, test_set_path, poi_position_path,
                      emb_size=emb_size_list[i],
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
                      device=device, writer=writer)

        # print_model_size(model) # show model parameters

        # show model hyper-parameters
        for param, value in model.hparams.items():
            logging.info(f"{param}: {value}")

        model_dir = './model'
        if mode == 'Train': # train model
            model.train_step()

            # show metrics
            logging.info(f'Best epoch:{model.best_epoch}\n{metrix2str(model.max_recall, model.max_mrr)}')
            recall_list.append(model.max_recall)
            mrr_list.append(model.max_mrr)

            with open('./model.py', 'r', encoding='utf-8') as file:
                model_py_content = file.read()
            writer.add_text('Model Source Code', model_py_content)

            metric = {}
            for N in [1, 5, 10, 20]:
                metric[f'Recall@{N}'] = model.max_recall[N - 1]
                metric[f'MRR@{N}'] = model.max_mrr[N - 1]

            writer.add_hparams(model.hparams, metric)
        else:# test model
            # load model
            log_str = '2024-01-08_03-54-32_FULL'
            model.load_state_dict(torch.load(
                os.path.join(model_dir, f'{log_str}_{dataset[i]}' + '.pth')))
            model.evaluate(-1, show_results=True)

        logger = logging.getLogger()
        logger.handlers[0].flush()
        logger.handlers[1].flush()
        writer.close()
        del model

        # break

    recall_array = np.array(recall_list)
    mrr_array = np.array(mrr_list)

    np.savez(os.path.join(log_path, f'{log_str}_metrics'), recall=recall_array, mrr=mrr_array)
    shutil.copyfile('./model.py', os.path.join(log_path, f'{log_str}_model.py'))
    shutil.copyfile('./config.py', os.path.join(log_path, f'{log_str}_config.py'))

    logging.info(f'Final results saved to local.')
    logging.info(f'4.Finish the training successfully!')
