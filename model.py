import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from dataset import *
from utils import *
from config import *
import logging
from tqdm import tqdm

FType = torch.FloatTensor
LType = torch.LongTensor

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device_str = "cpu"
if torch.cuda.is_available():
    device_str = "cuda:0"
elif torch.backends.mps.is_available():  # macos
    device_str = "mps:0"
device = torch.device(device_str)


class Model(nn.Module):
    def __init__(self, dataset_name, file_path_tr, file_path_te, poi_position_path,
                 emb_size=128, neg_size=10, hist_len=2,
                 user_count=992, item_count=5000,
                 learning_rate=0.001, decay=0.001, batch_size=1024,
                 epoch_num=100, top_n=30, num_workers=0, device=None, writer=None):
        super(Model, self).__init__()

        init_params = locals()
        allowed_types = (int, float, str, bool, torch.Tensor)
        self.hparams = {k: v for k, v in init_params.items() if isinstance(v, allowed_types)}

        self.device = device
        self.writer = writer

        self.dataset_name = dataset_name
        self.file_path_tr = file_path_tr
        self.file_path_tr = file_path_te

        self.user_count = user_count
        self.item_count = item_count
        self.node_count = user_count + item_count

        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.decay = decay
        self.batch_size = batch_size
        self.epochs = epoch_num

        self.top_n = top_n
        self.num_workers = num_workers

        # Node embedding
        self.emb = nn.Embedding(num_embeddings=self.node_count, embedding_dim=self.emb_size)

        # Inside exponential decay function interval:[0,1]
        self.time_decay = nn.Parameter(torch.sigmoid(torch.randn(self.node_count, dtype=torch.float)))
        self.pos_decay = nn.Parameter(torch.sigmoid(torch.randn(self.node_count, dtype=torch.float)))

        # Calculate long-short term weights
        self.linear_item = nn.Linear(in_features=self.emb_size, out_features=1, bias=True)
        self.linear_user = nn.Linear(in_features=self.emb_size, out_features=1, bias=True)

        self.optimizer = Adam(lr=self.lr, params=self.parameters(), weight_decay=self.decay)

        if generate_sample == True:
            self.data_tr = DataSetTrain(file_path_tr, user_count=self.user_count, item_count=self.item_count,
                                        neg_size=self.neg_size, hist_len=self.hist_len)
            self.data_te = DataSetTest(file_path_te, user_count=self.user_count, item_count=self.item_count,
                                       hist_len=self.hist_len)

            torch.save(self.data_tr, f'./processed_data/{dataset_name}_tr.pth')
            torch.save(self.data_te, f'./processed_data/{dataset_name}_te.pth')
        else:
            self.data_tr = torch.load(f'./processed_data/{dataset_name}_tr.pth')
            self.data_te = torch.load(f'./processed_data/{dataset_name}_te.pth')

        # Generate distance matrix
        self.all_dist = get_all_position(poi_position_path).to(self.device)  # item_num x item_num

        self.to(device)  # set parameter to same device

        # test results
        self.best_epoch = -1
        self.max_recall = np.zeros(self.top_n)
        self.max_mrr = np.zeros(self.top_n)

    def forward(self, s_nodes, h_nodes, h_times, h_masks, t_time, epoch=None):
        # h_nodes = h_nodes[:, -hist_length:] # control hist length
        # h_masks = h_masks[:, -hist_length:]
        # h_times = h_times[:, -hist_length:]

        s_node_embs = self.emb(s_nodes)  # user nodes: batch_size x emb_size
        h_node_embs = self.emb(h_nodes)  # history POI nodes: batch_size x hist_length x emb_size

        # combine long-short preference with weights
        short_pref_weight = torch.relu(self.linear_item(torch.mean(h_node_embs, dim=1)))  # batch_size x 1
        long_pref_weight = torch.relu(self.linear_user(s_node_embs))  # user personal preference: batch_size x 1
        all_weight = torch.softmax(torch.cat((short_pref_weight, long_pref_weight), dim=1), dim=1)
        short_pref_weight = all_weight[:, 0]  # batch_size x 1
        long_pref_weight = all_weight[:, 1]  # batch_size x 1

        self.time_decay.data.clamp_(min=1e-6)
        self.pos_decay.data.clamp_(min=1e-6)

        # time historical influence
        d_time = torch.abs(h_times - t_time.unsqueeze(dim=1))  # delta time: batch_size x hist_length
        d_time = torch.log(1 + d_time)  # reduce the difference between time interval
        h_time_decay = self.time_decay[s_nodes]  # user time decay weight: batch_size
        time_influence = torch.exp(
            torch.neg(d_time * h_time_decay.unsqueeze(dim=-1)))  # time influence: batch_size x hist_length

        # time_influence = time_influence.detach() * 0
        # print('no time')

        # position historical influence
        d_pos = self.all_dist[h_nodes, :]  # haversine distance: batch_size x hist_length x item_num
        h_pos_decay = self.pos_decay[s_nodes]  # user-item position decay weight: batch_size
        pos_influence = torch.exp(torch.neg( \
            d_pos * h_pos_decay.view(-1, 1, 1)))  # batch_size x hist_length x item_num

        # pos_influence = pos_influence.detach() * 0
        # print('no position')

        all_item_index = torch.arange(0, self.item_count).to(self.device)  # all POI nodes
        all_node_embs = self.emb(all_item_index)  # target POI nodes (waiting list): item_num x emb_size

        alpha_weights = torch.matmul(h_node_embs,
                                     all_node_embs.transpose(0, 1))  # batch_size x hist_length x item_num

        short_pref = (alpha_weights * (pos_influence + time_influence.unsqueeze(dim=-1)) * \
                      h_masks.unsqueeze(dim=-1)).mean(dim=1)

        # print('None pos or time')
        # short_pref = (alpha_weights * h_masks.unsqueeze(dim=-1)).mean(dim=1) # batch_size x item_num

        # user preference on target nodes (dot product): batch_size x item_num
        long_pref = torch.matmul(s_node_embs, torch.transpose(all_node_embs, 0, 1))

        # fusion all to get positive target rate
        all_scores = long_pref_weight.unsqueeze(dim=1) * long_pref + \
                     short_pref_weight.unsqueeze(dim=1) * short_pref  # batch_size x item_num

        # print('only long-term')
        # all_scores = long_pref_weight.unsqueeze(dim=1) * long_pref
        #
        # print('only short-term')
        # all_scores = short_pref_weight.unsqueeze(dim=1) * short_pref  # batch_size x item_num
        # print('no long-short weight')
        # all_scores =  long_pref + short_pref  # batch_size x item_num

        if self.writer != None and epoch != None:
            self.writer.add_scalar('model/position influence', pos_influence.mean().item(), epoch)
            self.writer.add_scalar('model/time influence', time_influence.mean().item(), epoch)

            self.writer.add_scalar('model/long-term weight', long_pref_weight.mean().item(), epoch)
            self.writer.add_scalar('model/long-term pref', long_pref.mean().item(), epoch)
            self.writer.add_scalar('model/short-term weight', short_pref_weight.mean().item(), epoch)
            self.writer.add_scalar('model/short-term pref', short_pref.mean().item(), epoch)

        return all_scores  # batch_size x item_num

    def cal_loss(self, p_scores, n_scores=None):
        epsilon = 1e-6
        loss = -torch.log(epsilon + torch.sigmoid(p_scores)).sum()
        if n_scores != None:
            neg_loss = -torch.log(epsilon + 1 - torch.sigmoid(n_scores)).sum()
            loss += neg_loss

        return loss

    def train_step(self):
        self.train()  # Set the model to training mode
        self.to(self.device)

        loader = DataLoader(self.data_tr, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        for epoch in range(self.epochs):
            count_all = 0
            total_loss = 0

            progress_bar = tqdm(enumerate(loader), total=len(loader),
                                desc=f"Epoch {epoch + 1}/{self.epochs}")

            if self.writer != None:
                for name, param in self.named_parameters():
                    self.writer.add_scalar('parameters/' + name, param.mean().item(), epoch)
                    if param.grad is not None:
                        self.writer.add_scalar('gradients/' + name, param.grad.mean().item(), epoch)
                # self.writer.add_embedding(self.emb.weight.data.cpu(), global_step=epoch)

            with torch.no_grad():
                for i_batch, sample_batched in enumerate(loader):
                    all_scores = self.forward(sample_batched['source_node'].type(LType).to(self.device),
                                              sample_batched['history_nodes'].type(LType).to(self.device),
                                              sample_batched['history_times'].type(FType).to(self.device),
                                              sample_batched['history_masks'].type(FType).to(self.device),
                                              sample_batched['target_time'].type(FType).to(self.device),
                                              epoch=epoch)

                    positive_targets = sample_batched['target_node'].type(LType).to(self.device)  # batch_size
                    negative_targets = sample_batched['neg_nodes'].type(LType).to(self.device)  # batch_size x neg_num

                    batch_size, num_negatives = negative_targets.size()
                    pos_scores = all_scores[torch.arange(batch_size), positive_targets].unsqueeze(1)
                    neg_scores = all_scores[torch.arange(batch_size).unsqueeze(1), negative_targets]

                    if self.writer != None:
                        self.writer.add_scalar('model/positive scores', pos_scores.mean().item(), epoch)
                        self.writer.add_scalar('model/negative scores', neg_scores.mean().item(), epoch)

                    break

            self.train()
            for i_batch, sample_batched in progress_bar:
                self.optimizer.zero_grad()

                all_scores = self.forward(sample_batched['source_node'].type(LType).to(self.device),
                                          sample_batched['history_nodes'].type(LType).to(self.device),
                                          sample_batched['history_times'].type(FType).to(self.device),
                                          sample_batched['history_masks'].type(FType).to(self.device),
                                          sample_batched['target_time'].type(FType).to(
                                              self.device))  # batch_size x item_num

                positive_targets = sample_batched['target_node'].type(LType).to(self.device)  # batch_size
                negative_targets = sample_batched['neg_nodes'].type(LType).to(self.device)  # batch_size x neg_num

                batch_size, num_negatives = negative_targets.size()
                pos_scores = all_scores[torch.arange(batch_size), positive_targets].unsqueeze(1)
                neg_scores = all_scores[torch.arange(batch_size).unsqueeze(1), negative_targets]

                loss = self.cal_loss(pos_scores, neg_scores)
                # loss = self.cal_loss(pos_scores)

                loss.backward()  # Backward pass
                self.optimizer.step()  # Update the weights

                total_loss += loss.item()
                count_all += len(sample_batched['target_node'])
                progress_bar.set_postfix(loss=loss.item())

            self.evaluate(epoch, show_results=False)

            if self.writer != None:
                self.writer.add_scalar('loss/train', loss.item(), epoch)
                self.writer.flush()

        return self.max_recall, self.max_mrr

    def evaluate(self, epoch, show_results=False):
        self.eval()  # Set the model to evaluation mode
        self.to(device)

        count_all = 0
        recall_all = np.zeros(self.top_n, dtype=float)
        MRR_all = np.zeros(self.top_n, dtype=float)

        with torch.no_grad():
            total_loss = 0

            loader = DataLoader(self.data_te, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

            for i_batch, sample_batched in enumerate(loader):
                # ranking scores: batch_size x item_num
                all_scores = self.forward(sample_batched['source_node'].type(LType).to(self.device),
                                          sample_batched['history_nodes'].type(LType).to(self.device),
                                          sample_batched['history_times'].type(FType).to(self.device),
                                          sample_batched['history_masks'].type(FType).to(self.device),
                                          sample_batched['target_time'].type(FType).to(self.device))

                # testing loss
                targets = sample_batched['target_node'].type(LType).to(self.device)
                target_scores = all_scores[torch.arange(all_scores.size(0)).to(self.device), targets].data.cpu()

                loss = self.cal_loss(target_scores)
                total_loss += loss.cpu().item()

                sorted_poi_scores = torch.argsort(all_scores, dim=1, descending=True)  # batch_size x item_num
                _, col_index = torch.where(sorted_poi_scores == \
                                           sample_batched['target_node'].type(LType).to(self.device).unsqueeze(
                                               dim=-1))  # batch_size

                col_index = col_index[col_index <= self.top_n - 1]

                col_index = col_index.cpu().numpy().tolist()
                recall = np.zeros(self.top_n, dtype=float)
                MRR = np.zeros(self.top_n, dtype=float)

                for target_index in col_index:
                    recall[target_index:] += 1
                    MRR[target_index:] += 1. / (target_index + 1)

                recall_all += recall
                MRR_all += MRR
                count_all += len(sample_batched['target_node'])

            if self.writer != None:
                self.writer.add_scalar('loss/validate', loss.item(), epoch)

        recall_all = recall_all / count_all
        MRR_all = MRR_all / count_all

        # update the metrics during training
        if recall_all.mean() > self.max_recall.mean() and \
                recall_all[9] > self.max_recall[9] and \
                recall_all[19] > self.max_recall[19] and \
                epoch != -1:
            self.best_epoch = epoch
            self.max_recall = recall_all
            self.max_mrr = MRR_all
            torch.save(self.state_dict(), f'./model/{log_str}_{self.dataset_name}.pth')

        if self.writer != None:
            for N in topN:
                self.writer.add_scalar(f'Recall/top@{N}', recall_all[N - 1], epoch)
                self.writer.add_scalar(f'MRR/top@{N}', MRR_all[N - 1], epoch)

        if epoch == -1:  # only evaluate
            logging.info(f'\n Evaluate:\n{metrix2str(recall_all, MRR_all)}')

        if show_results == True:  # output metrics
            logging.info(f'\nTesting epoch: {epoch + 1}\n{metrix2str(recall_all, MRR_all)} \n \
                        \nBest epoch:{self.best_epoch + 1}\n{metrix2str(self.max_recall, self.max_mrr)}')

        return [recall_all, MRR_all]


if __name__ == '__main__':
    torch.manual_seed(42)

    create_log('time_pos')
    for i in [0, 1, 2]:
        logging.info(f'+++++++++++++ Dataset: {dataset[i]} ++++++++++++++++')

        train_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_train.txt')
        test_set_path = os.path.join(save_path, dataset[i], dataset[i] + '_test.txt')
        poi_position_path = os.path.join(save_path, dataset[i], dataset[i] + '_POI_position.txt')

        assert os.path.exists(train_set_path)
        assert os.path.exists(test_set_path)

        htne = Model(dataset[i], train_set_path, test_set_path, poi_position_path,
                     emb_size=emb_size_list[i],
                     neg_size=neg_size_list[i],
                     hist_len=hist_length_list[i],
                     user_count=user_cnt_list[i],
                     item_count=poi_cnt_list[i],
                     learning_rate=learning_rate_list[i],
                     decay=decay_list[i],
                     batch_size=batch_size_list[i],
                     test_and_save_step=10,
                     epoch_num=epoch_num_list[i],
                     top_n=20,
                     num_workers=0,
                     device=device)
        print_model_size(htne)

        # torch.save(htne.state_dict(), './untrained_parameters.pth')
        # model.load_state_dict(torch.load('./untrained_parameters.pth'))
        #
        # torch.save(htne, './untrained_model.pth')
        # model.load('./untrained_model.pth')

        htne.evaluate(-1)
        htne.train_step()
        # break
    logging.info('Finish all training!')
