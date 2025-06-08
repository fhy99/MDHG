import time
from util import Data
from model import *
import os
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='dataset name: retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.005, help='ssl task maginitude')
parser.add_argument('--lam', type=float, default=0.0001, help='diff task maginitude')
parser.add_argument('--eps', type=float, default=0.2, help='eps')
parser.add_argument('--K1', type=int, default=80, help='numbers')
parser.add_argument('--K2', type=int, default=50, help='numbers')
parser.add_argument('--K3', type=int, default=20, help='numbers')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.set_device(opt.gpu_id)
# torch.device('cpu')

def reset_parameters(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.init.xavier_uniform_(layer.weight)

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    #exp_seed = 2023
    #init_seed(exp_seed)
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    if opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'retailrocket':
        n_node = 36968
    elif opt.dataset == 'amazon':
        n_node = 18888
    else:
        n_node = 309
    train_data = Data(train_data,all_train, shuffle=False, n_node=n_node)
    test_data = Data(test_data,all_train, shuffle=False, n_node=n_node)
    model = trans_to_cuda(MDHG(R = train_data.R,adj1 = train_data.adj1, adj2 = train_data.adj2,adjacency=train_data.adjacency,adjacency_T=train_data.adjacency_T,adjacency1=train_data.adjacency1,R1 = train_data.R1,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta,lam= opt.lam,eps=opt.eps,layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset,K1=opt.K1,K2=opt.K2,K3=opt.K3,dropout=opt.dropout,alpha=opt.alpha))
    reset_parameters(model)
    top_K = [5, 10, 20, 50]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        # model.e_step()
        metrics, total_loss = train_test(model, train_data, test_data, epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tNDCG%d: %.4f\tEpoch: %d,  %d, %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   K, best_results['metric%d' % K][2], best_results['epoch%d' % K][0], best_results['epoch%d' % K][1],
                   best_results['epoch%d' % K][2]))


if __name__ == '__main__':
    main()
