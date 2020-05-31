import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
import pickle
import argparse
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from continue_learning.continuum import Continuum
from continue_learning.continue_graphsage import IncrementSupervisedGraphSage

def run_pubmed_incremental(feat_data, labels, adj_lists, args):
    evaluation_metrics = []
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = IncrementSupervisedGraphSage(3, enc2, labels, args)
    val_data = Continuum(name="pubmed", data_type='val', download=True)
    
    val = val_data.nodes()
    for t in range(val_data.num_class):
        incremental_data = Continuum(name="pubmed", data_type='train', download=True, task_type=t)  
        train = incremental_data.nodes()
        random.shuffle(train)
        print("the size of task: %i"%len(train))
        for i in range(0, len(train), args.batch_size):
            if i+args.batch_size <= len(train):
                batch_nodes = train[i:i+args.batch_size]
            else:
                batch_nodes = train[i:len(train)]
            graphsage.observe(batch_nodes)

        val_output = graphsage.forward(val) 
        evaluation_metrics.append([i, len(train), f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")])
    return evaluation_metrics

def run_cora(feat_data, labels, adj_lists,args):
    evaluation_metrics = []
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = IncrementSupervisedGraphSage(7, enc2, labels, args)

    val_data = Continuum(name="cora", data_type='val', download=True)
    val = val_data.nodes()
    incremental_data = Continuum(name="cora", data_type='all_train', download=True)  
    train = incremental_data.nodes()
    random.shuffle(train)
    for i in range(0, len(train), args.batch_size):
        if i+args.batch_size <= len(train):
            batch_nodes = train[i:i+args.batch_size]
        else:
            batch_nodes = train[i:len(train)]
        graphsage.observe(batch_nodes)

        val_output = graphsage.forward(val) 
        evaluation_metrics.append([i, len(train), f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")])
    return evaluation_metrics

def run_cora_incremental(feat_data, labels, adj_lists, args):
    evaluation_metrics = []
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = IncrementSupervisedGraphSage(7, enc2, labels, args)

    val_data = Continuum(name=args.dataset, data_type='val', download=True)
    val = val_data.nodes()
    for i in range(7):
        incremental_data = Continuum(name=args.dataset, data_type='train', download=True, task_type=i)  
        train = incremental_data.nodes()
        random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            if i+args.batch_size <= len(train):
                batch_nodes = train[i:i+args.batch_size]
            else:
                batch_nodes = train[i:len(train)]
            graphsage.observe(batch_nodes)

        val_output = graphsage.forward(val)
        train_output = graphsage.forward(train)
        #print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        evaluation_metrics.append([i, len(train), f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")])
    return evaluation_metrics
                       
parser = argparse.ArgumentParser(description='Feature Graph Networks')
parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset location")
parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, or pubmed")
parser.add_argument("--lr", type=float, default=0.7, help="learning rate")
parser.add_argument("--batch-size", type=int, default=10, help="minibatch size")
parser.add_argument("--iteration", type=int, default=5, help="number of training iteration")
parser.add_argument("--memory-size", type=int, default=100, help="number of samples")
parser.add_argument("--momentum", type=float, default=0, help="momentum of SGD optimizer")
parser.add_argument("--adj-momentum", type=float, default=0.9, help="momentum of the feature adjacency")
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--repeat', type=int, default=10, help='repeat experiment.')
parser.add_argument("-s", "--save", action="store_true", help="increase output verbosity")
parser.add_argument("-p", "--plot", action="store_true", help="plot the evaluation result")
parser.add_argument("-t", "--task_incremental", action="store_false", help="continue learning")
args = parser.parse_args(); print(args)
torch.manual_seed(args.seed)


if __name__ == "__main__":
    total_results = []
    for memory in [50,100,200,500]:
        args.memory_size = memory
        results = []
        for i in range(args.repeat):
            args.seed = i
            print(args)
            np.random.seed(args.seed)
            random.seed(args.seed)
            incremental_data = Continuum(name=args.dataset, data_type='all_train', download=True, task_type = 0)
            num_nodes = len(incremental_data.labels)

            adj_lists = incremental_data.neighbors()
            feat_data = incremental_data.features.numpy()
            labels = incremental_data.labels
            if args.dataset == "pubmed":
                results.append(run_pubmed_incremental(feat_data, labels, adj_lists, args))
            if args.task_incremental and args.dataset == "cora": 
                results.append(run_cora_incremental(feat_data, labels, adj_lists, args))
            elif args.dataset == "cora": 
                results.append(run_cora(feat_data, labels, adj_lists, args))
        total_results.append(results)
    total_results = np.array(total_results)
    
    if args.plot:
        print(total_results.shape)
        tasks = np.arange(total_results.shape[2])
        plt.plot(tasks, total_results[...,:,2].mean(axis=1)[4],"k-^", label = "memory_size: 1000")
        plt.plot(tasks, total_results[...,:,2].mean(axis=1)[3],"r-o", label = "memory_size: 500")
        plt.plot(tasks, total_results[...,:,2].mean(axis=1)[2],"g-x", label = "memory_size: 200")
        plt.plot(tasks, total_results[...,:,2].mean(axis=1)[1],"b-+", label = "memory_size: 100")
        plt.plot(tasks, total_results[...,:,2].mean(axis=1)[0],"y-*", label = "memory_size: 50")

        plt.title("datasets: %s iteration: %s lr: %s batch_szie: %s"%\
                  (args.dataset,args.iteration, args.lr, args.batch_size))
        plt.legend()
        plt.xlabel("task")
        plt.ylabel("accuracy (%)")
        plt.savefig("doc/task_incre_plt_"+args.dataset+".png")
        
    if args.save:
        save_file = "doc/save-%f"%time.time()
        with open(save_file,"wb") as file:
            pickle.dump(total_results, file)
        np.savetxt(save_file+args.dataset+".txt",total_results[...,:,2].mean(axis=1),fmt="%2.3f")
        with open(save_file+args.dataset+"_perm"+".txt","w") as file:
            file.write(str(args))
            