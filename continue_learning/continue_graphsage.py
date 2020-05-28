import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import random

class IncrementSupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, labels, args, memory_size=100):
        super(IncrementSupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.memory_size = memory_size
        self.memory_nodes = torch.empty(0)
        self.labels = labels
        self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.parameters()), lr=0.7)
    
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)
        self.args = args

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()
    
    def observe(self, nodes):
        for batch in range(self.args.iteration):
            self.optimizer.zero_grad()
            loss = self.loss(nodes, Variable(torch.LongTensor(self.labels[np.array(nodes)])))
            loss.backward()
        self.optimizer.step()
        self.sample(nodes)
        
        minibatches = [self.memory_nodes[i:i+self.args.batch_size] for i in range(0,self.memory_nodes.shape[0], self.args.batch_size)]
        for batch in minibatches:
            self.optimizer.zero_grad()
            loss = self.loss(batch, Variable(torch.LongTensor(self.labels[batch])))
            loss.backward()
            self.optimizer.step()
        print(loss.data.item())
    
    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
    
    @torch.no_grad()
    def sample(self, nodes):
#         self.minibatch_nodes = self.minibatch_nodes.union(set(nodes))\
        self.memory_nodes = torch.LongTensor(np.union1d(self.memory_nodes, nodes))
        if self.memory_nodes.shape[0]> self.memory_size:
            self.memory_nodes = self.memory_nodes[torch.randperm(self.memory_size)]# random selection
            