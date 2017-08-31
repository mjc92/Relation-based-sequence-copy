import torch
import numpy as np
import argparse
from collections import Counter
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def pack_padded(outputs, targets):
    """
    outputs: Variable, [b x seq x vocab]
    targets: Variable, [b x seq]
    """
    out_pack = []
    tar_pack = []
    target_lens = (targets>0).long().sum(1).data
    for i,length in enumerate(target_lens):
        out_pack.append(outputs[i,:length])
        tar_pack.append(targets[i,:length])
    return torch.cat(out_pack,0), torch.cat(tar_pack,0)