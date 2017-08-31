import torch
import os
from torch.utils import data
import numpy as np

"""
Sample usage case
src_root = '/home/irteam/users/mjchoi/github/DL2DL/data/task1/source/'
trg_root = '/home/irteam/users/mjchoi/github/DL2DL/data/task1/target/'
data_loader = get_loader(src_root, trg_root, 2)
src,trg, src_len, trg_len = dat_iter.next()
"""

class TextFolder(data.Dataset):
    def __init__(self, root, ptr, max_out):
        """
        Initializes paths and preprocessing module
        root: data directory
        ptr: whether looking for pointer or value
        max_out: max length of output (when using pointer)
        """
        self.ptr = ptr
        self.max_out = max_out
        with open(root) as f:
            self.data = f.read().split('\n')
        
    def __getitem__(self, index):
        data = self.data[index].split(':')
        src1 = [int(x) for x in data[0].split(' ')]
        src2 = [int(x) for x in data[1].split(' ')]
        trg1 = [int(x) for x in data[2].split(' ')]
        trg2 = [int(x) for x in data[3].split(' ')]
        
        # indices for cross entropy
        if self.ptr:
            idxs = [(trg1+src2).index(x) for x in trg2] + [self.max_out-1] # replaces EOS
        else:
            idxs = trg2.copy()
            trg2 = [2] + trg2 + [3]
        # sos + trg + eos
        src1 = torch.LongTensor(src1)
        trg1 = torch.LongTensor(trg1)
        src2 = torch.LongTensor(src2)
        trg2 = torch.LongTensor(trg2)
        idxs = torch.LongTensor(idxs)
        return src1, trg1, src2, trg2, idxs
        # return torch.LongTensor(src1), torch.LongTensor(trg1), \
        #     torch.LongTensor(src2), torch.LongTensor(trg2), torch.LongTensor(idxs)
        # return torch.LongTensor(src1), torch.LongTensor(trg1),\
        #     torch.LongTensor(src2), torch.LongTensor(trg2)

    def __len__(self):
        return len(self.data)
        
def flatten(self, listoflist):
    function = lambda l: [item for sublist in l for item in sublist]
    return function(listoflist)

def collate_fn(data):
    # Sort function: sorts in decreasing order by the length of the items in the right (targets)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    src1, trg1, src2, trg2, idxs  = zip(*data)
    
    """
    -- Inputs --
    src1, trg1, src2 : list of lists, inputs
    trg2 : list of lists, output
    idxs : list of lists, index for each token in trg2
    
    -- Outputs --
    1. context_len: a list of len batch, lengths of all contexts (mostly 10)
    2. sources_out: a tensor of size [batch*10-a x seq], all input sentences merged into 1 matrix
    3. source_len: a list of len batch*10-a, length of every line in 1
    4. queries_out: a tensor of size [batch x seq], all queries merged into 1 matrix
    5. query_len: a list of len batch, lengths of all queries
    6. targets_out: a tensor of size [batch x seq], all outputs merged into 1 matrix
    7. target_len: a list of len batch, lenghts of all answers
    """
    
    src_list = []
    src_len = []
    trg_list = []
    trg_len = []
    label_list = []
    label_len = []
    for (s1,t1,s2,t2,idx) in zip(src1,trg1,src2,trg2,idxs):
        src_list.append(torch.cat([t1,s2],0))
        src_len.append(len(src_list[-1]))
        trg_list.append(t2)
        trg_len.append(len(trg_list[-1]))
        label_list.append(idx)
        label_len.append(len(label_list[-1]))
    
    sources_out = torch.LongTensor(len(src_list),max(src_len)).zero_()
    targets_out = torch.LongTensor(len(trg_list),max(trg_len)).zero_()
    labels_out  = torch.LongTensor(len(label_list),max(label_len)).zero_()
    
    for i in range(len(src_list)):
        sources_out[i,:src_len[i]] += src_list[i]
        targets_out[i,:trg_len[i]] += trg_list[i]
        labels_out[i,:label_len[i]] += label_list[i]
    return sources_out, targets_out,labels_out
    

def get_loader(root, ptr, batch_size=64, max_out=30, num_workers=2, shuffle=True):
    dataset = TextFolder(root, ptr, max_out)
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader