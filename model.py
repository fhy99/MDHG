import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from numba import jit
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class ItemConv(Module):
    def __init__(self, layers, K1, K2, K3, dropout, alpha, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = {}
        self.w_i1 = {}
        self.w_i2 = {}
        self.dropout = nn.Dropout(p=dropout)
        self.k1 = K1
        self.k2 = K2
        self.k3 = K3
        self.channel = 3
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(alpha)

        for i in range(self.channel):
            self.w_i1['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

        #self.w_i1 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.w_i2['weight_item0'] = nn.Linear(self.emb_size, self.k1, bias=False)
        self.w_i2['weight_item1'] = nn.Linear(self.emb_size, self.k2, bias=False)
        self.w_i2['weight_item2'] = nn.Linear(self.emb_size, self.k3, bias=False)
        self.w_i3 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))


    def forward(self, adj, adjacency, embedding, channel):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        finalh = []
    
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % i])(item_embeddings)
            item_embeddings = torch.sparse.mm(adjacency, item_embeddings)

            H1 = trans_to_cuda(self.w_i1['weight_item%d' % channel])(item_embeddings) + item_embeddings
            H1 = self.relu(H1)
            H1 = trans_to_cuda(self.w_i2['weight_item%d' % channel])(H1)
            H1 = torch.softmax(H1, dim=1)  # shape: [n_node, K]

            h = H1.T.mul(adj)  # shape: [n_node, K]
            h = h.mul(1.0 / torch.sum(h, dim=0))
        
            h = h @ item_embeddings  # shape: [K, emb_size]
            h = H1 @ h  # shape: [n_node, emb_size]
            item_embeddings = h + item_embeddings
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
            finalh.append(F.normalize(h, dim=-1, p=2))

        item_embeddings = torch.sum(torch.stack(final), 0) / (self.layers + 1)
        hs = torch.sum(torch.stack(finalh), 0) / (self.layers)
        return item_embeddings, hs

class MDHG(Module):
    def __init__(self, R, adj1, adj2, adjacency, adjacency_T, adjacency1, R1, n_node, lr, layers, l2, beta,lam,eps, dataset, K1, K2, K3, dropout, alpha, emb_size=100, batch_size=100):
        super(MDHG, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.alpha = alpha
        self.w_k = 10
        self.num = 5000
        self.adjacency = trans_to_cuda(self.trans_adj(adjacency))
        self.adjacency_T = trans_to_cuda(self.trans_adj(adjacency_T))
        self.adjacency1 = trans_to_cuda(self.trans_adj(adjacency1))
        self.adj1 = torch.cuda.FloatTensor(adj1)
        self.adj2 = torch.cuda.FloatTensor(adj2)
        self.R = torch.cuda.FloatTensor(R)
        self.R1 = torch.cuda.FloatTensor(R1)
        self.lamb = 0.0001
        self.embedding1 = nn.Embedding(self.n_node, self.emb_size)
        self.embedding2 = nn.Embedding(self.n_node, self.emb_size)
        self.embedding3 = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200
        if self.dataset == 'retailrocket':
            self.pos_len = 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.ItemGraph = ItemConv(self.layers, self.K1, self.K2, self.K3, dropout, self.alpha)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_11 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.attn = nn.Parameter(torch.Tensor(1, self.emb_size))

        self.weights = {}
        self.attention = nn.Parameter(torch.Tensor(1, self.emb_size))
        self.attention_mat = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))

        self.Wa = nn.Parameter(torch.Tensor(10, self.emb_size + 1, self.emb_size))
        self.Wb = nn.Parameter(torch.Tensor(10, self.emb_size + 1, self.emb_size))
        self.Wc = nn.Parameter(torch.Tensor(10, self.emb_size + 1, self.emb_size))
        self.Wf = nn.Parameter(torch.Tensor(1, 10))
        self.bias = nn.Parameter(torch.Tensor(1, 100))

        self.w_hh = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))

        self.adv_item = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        self.adv_sess = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        self.loss_function = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def trans_adj(self, adjacency):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return adjacency

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):

        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)

        return select

    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select


    def cross_view(self, hidden1, hidden2, hidden3):

        channel_embeddings = [hidden1, hidden2, hidden3]
        weights = []
        for embedding in channel_embeddings:
            weights.append(torch.sum(
                torch.mul(self.attention, torch.matmul(embedding, self.attention_mat)), 1))
        weights = torch.stack((weights[0],weights[1],weights[2]),dim=0)

        score = torch.softmax(torch.transpose(weights, 1, 0), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.transpose(torch.mul(torch.transpose(score, 1, 0)[i], torch.transpose(channel_embeddings[i], 1, 0)), 1, 0)
        return mixed_embeddings, score

    def write_to_file(self, recommended_items, filename):
        with open(filename, 'a') as file:
            for session in recommended_items:
                file.write(','.join(map(str, session.tolist())) + '\n')

    def forward(self, session_item, session_len, reversed_sess_item, mask, epoch, tar, train):

        def score(x1, x2):
            # return torch.sum(torch.mul(x1, x2), dim=1)
            return torch.mean((x1 - x2) ** 2)

        def row_shuffle(x, size):
            id = torch.randperm(size)
            return x[id, :].view(x.size())

        if train:
            
            item_embeddings_1, h_1 = self.ItemGraph(self.adj1, self.adjacency, self.embedding1.weight, 0)
            item_embeddings_2, h_2 = self.ItemGraph(self.adj2, self.adjacency_T, self.embedding2.weight, 1)
            item_embeddings_3, h_3 = self.ItemGraph(self.R1, self.adjacency1, self.embedding3.weight, 2)
            
            item_embeddings_1 = F.normalize(item_embeddings_1, dim=-1, p=2)
            item_embeddings_2 = F.normalize(item_embeddings_2, dim=-1, p=2)
            item_embeddings_3 = F.normalize(item_embeddings_3, dim=-1, p=2)
            
            item_embeddings_i, self.scores = self.cross_view(item_embeddings_1, item_embeddings_2, item_embeddings_3)

            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len,reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)
            # print(loss_item.item())

            h_1 = h_1[session_item - 1]
            h_2 = h_2[session_item - 1]
            h_3 = h_3[session_item - 1]

            pos = score(h_1, h_2)
            neg1 = score(row_shuffle(h_1, self.batch_size), h_2)
            local_loss1 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
            pos = score(h_2, h_3)
            neg2 = score(row_shuffle(h_2, self.batch_size), h_3)
            local_loss2 = torch.sum(-torch.log(torch.sigmoid(pos - neg2)))
            pos = score(h_1, h_3)
            neg3 = score(row_shuffle(h_1, self.batch_size), h_3)
            local_loss3 = torch.sum(-torch.log(torch.sigmoid(pos - neg3)))

            con_loss = self.lam * (local_loss1 + local_loss2 + local_loss3)

        else:
            item_embeddings_1, h_1 = self.ItemGraph(self.adj1, self.adjacency, self.embedding1.weight, 0)
            item_embeddings_2, h_2 = self.ItemGraph(self.adj2, self.adjacency_T, self.embedding2.weight, 1)
            item_embeddings_3, h_3 = self.ItemGraph(self.R1, self.adjacency1, self.embedding3.weight, 2)
            
            item_embeddings_1 = F.normalize(item_embeddings_1, dim=-1, p=2)
            item_embeddings_2 = F.normalize(item_embeddings_2, dim=-1, p=2)
            item_embeddings_3 = F.normalize(item_embeddings_3, dim=-1, p=2)

            item_embeddings_i, self.scores = self.cross_view(item_embeddings_1, item_embeddings_2, item_embeddings_3)

            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len,
                                                         reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item,
                                                    mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)

            con_loss = 0
        return con_loss, loss_item, scores_item, 0


def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    con_loss, loss_item, scores_item, loss_diff = model(session_item, session_len, reversed_sess_item, mask, epoch,tar, train)
    return tar, scores_item, con_loss, loss_item, loss_diff

@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids#,k_largest_scores


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        model.zero_grad()
        tar, scores_item, con_loss, loss_item, loss_diff = forward(model, i, train_data, epoch, train=True)
        loss = loss_item + con_loss + loss_diff
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20, 50]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        tar, scores_item, con_loss, loss_item, loss_diff = forward(model, i, test_data, epoch, train=False)
        scores = trans_to_cpu(scores_item).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(50, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                prediction_list = prediction.tolist()
                epsilon = 0.1 ** 10
                DCG = 0
                IDCG = 0
                for j in range(K):
                    if prediction_list[j] == target:
                        DCG += 1 / math.log2(j + 2)
                for j in range(min(1, K)):
                    IDCG += 1 / math.log2(j + 2)
                metrics['ndcg%d' % K].append(DCG / max(IDCG, epsilon))
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss




