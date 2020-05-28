import os
import dgl
import tqdm
import torch
import os.path
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
from dgl.data import citegrh
from itertools  import compress
from torchvision.datasets import VisionDataset
torch.utils.data.TensorDataset

from sklearn.metrics import f1_score
from collections import defaultdict

class Continuum(VisionDataset):
    def __init__(self, root='~/.dgl', name = 'cora', data_type='train', download=True, task_type = 0):
        super(Continuum, self).__init__(root)
        self.name = name

        self.download()
        self.features = torch.FloatTensor(self.data.features)
        self.ids = torch.LongTensor(list(range(self.features.size(0))))
        graph = DGLGraph(self.data.graph)
        graph = dgl.transform.add_self_loop(graph)
        self.src, self.dst = graph.edges()
        self.labels = torch.LongTensor(self.data.labels)

        if data_type == 'train':
            mask = np.logical_or(self.data.test_mask,self.data.train_mask)
            self.mask = (np.logical_and((self.labels==task_type),mask)).type(torch.bool)#low efficient
        elif data_type == 'val':
            self.mask = torch.BoolTensor(self.data.val_mask)
        elif data_type == 'test':
            self.mask = torch.BoolTensor(self.data.test_mask)
        elif data_type == "all_train":
            self.mask = np.logical_or(self.data.test_mask,self.data.train_mask)
            
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        neighbor = self.features[self.dst[self.src==self.ids[self.mask][index]]]
        return self.features[self.mask][index].unsqueeze(-2), self.labels[self.mask][index], neighbor.unsqueeze(-2)
    
    def neighbors(self):
        adj_lists = defaultdict(set)
        for i in range(self.src.size(0)):
            paper1 = self.src[i].item()
            paper2 = self.dst[i].item()
            if not paper1 == paper2:
                adj_lists[paper1].add(paper2)
        return adj_lists
    
    def nodes(self):
        return self.ids[self.mask].numpy()
    
    def download(self):
        """Download data if it doesn't exist in processed_folder already."""
        processed_folder = os.path.join(self.root, self.name)
        os.makedirs(processed_folder, exist_ok=True)
        os.environ["DGL_DOWNLOAD_DIR"] = processed_folder
        data_file = os.path.join(processed_folder, 'data.pt')
        if os.path.exists(data_file):
            self.data = torch.load(data_file)
        else:
            if self.name.lower() == 'cora':
                self.data = citegrh.load_cora()
            elif self.name.lower() == 'citeseer':
                self.data = citegrh.load_citeseer()
            elif self.name.lower() == 'pubmed':
                self.data = citegrh.load_pubmed()
            else:
                raise RuntimeError('Citation dataset name {} wrong'.format(self.name))
            with open(data_file, 'wb') as f:
                torch.save(self.data, data_file)
        self.feat_len, self.num_class = self.data.features.shape[1], self.data.num_labels
