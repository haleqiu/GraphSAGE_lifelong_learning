import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import random

class IncrementSupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, labels, args):
        super(IncrementSupervisedGraphSage, self).__init__()
        self.args = args
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.memory_size = self.args.memory_size
        self.memory_nodes = torch.LongTensor()
        self.memory_order = torch.LongTensor()
        self.sample_viewed = 0
        self.labels = labels
        self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.parameters()), lr=0.7)
    
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()
    
    def observe(self, nodes):
        for itera in range(self.args.iteration):
            if nodes.size ==1:
                break
            self.optimizer.zero_grad()
            loss = self.loss(nodes, Variable(torch.LongTensor(self.labels[np.array(nodes)])))
            loss.backward()
        self.optimizer.step()
        self.sample(nodes)
        minibatches = [self.memory_nodes[i:i+self.args.batch_size] for i in range(0,self.memory_nodes.shape[0], self.args.batch_size)]
        random.shuffle(minibatches)
        for batch in minibatches:
            if len(batch) ==1:
                continue
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
        nodes = torch.LongTensor(nodes)
        self.sample_viewed += nodes.size()[0]
        self.memory_order += nodes.size()[0]
        self.memory_nodes = torch.cat((self.memory_nodes,nodes.type_as(self.memory_nodes)), dim = 0)
        self.memory_order = torch.cat((self.memory_order,torch.LongTensor(list(range(nodes.size()[0]-1,-1,-1)))), dim = 0)# for debug
        node_len = int(self.memory_nodes.shape[0])
        ext_memory = node_len - self.memory_size
        if ext_memory > 0:
            mask = torch.zeros(node_len,dtype = bool)
            reserve = self.memory_size #reserved memrory to be stored
            seg = np.append(np.arange(0,self.sample_viewed,self.sample_viewed/ext_memory),self.sample_viewed)
            for i in range(len(seg)-2,-1,-1):
                left = self.memory_order.ge(np.ceil(seg[i]))*self.memory_order.lt(np.floor(seg[i+1]))
                leftindex = left.nonzero()
                if leftindex.size()[0] > reserve/(i+1):#the quote is not enough, need to be reduced
                    leftindex = leftindex[torch.randperm(leftindex.size()[0])[:int(reserve/(i+1))]]#reserve the quote
                    mask[leftindex] = True
                else:
                    mask[leftindex] = True #the quote is enough
                reserve -= leftindex.size()[0]#deducte the quote
            self.memory_nodes = self.memory_nodes[mask]
            self.memory_order = self.memory_order[mask]

    @torch.no_grad()
    def uniform_sample(self, nodes):
        #a naive random sampling
        nodes = torch.LongTensor(nodes)
        self.memory_nodes = torch.cat((self.memory_nodes, nodes),dim=0)
        if self.memory_nodes.shape[0]> self.memory_size:
            self.memory_nodes = self.memory_nodes[torch.randperm(self.memory_size)]# random selection

