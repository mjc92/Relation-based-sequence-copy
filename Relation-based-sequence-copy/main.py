import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import datetime
import torch
from torch import optim, nn
import argparse
import os
import torch.nn.functional as F
from torch.autograd import Variable
from packages.data_loader import get_loader
from packages.functions import str2bool, pack_padded

parser = argparse.ArgumentParser()

parser.add_argument('--train_root',type=str,default='data/1case/train.txt',help='data file')
parser.add_argument('--test_root',type=str,default='data/1case/test.txt',help='data file')
parser.add_argument('--vocab',type=int,default=100,help='vocab size')

parser.add_argument("--mode",type=str, help='train/test mode. Error if unspecified')
parser.add_argument("--epochs",type=int, default=100, help='Number of epochs. Set by default to 10')
parser.add_argument("--lr",type=float, default=0.01, help='learning rate')
parser.add_argument("--batch",type=int, default=64, help='batch size')
parser.add_argument("--cuda",type=str2bool, default=False, help='whether to use cuda')

# arguments related to the model structure itself
parser.add_argument("--ptr", type=str2bool, default=True, help='pointer network if True')
parser.add_argument("--hidden",type=int, default=256, help='size of hidden dimension')
parser.add_argument("--embed",type=int, default=256, help='size of embedded word dimension')
parser.add_argument("--n_layers",type=int, default=1, help='number of layers for transformer model')
parser.add_argument("--n_head",type=int, default=8, help='number of heads for transformer model')
# parser.add_argument("--max_in_seq",type=int, default=100, help='max length of input')
parser.add_argument("--max_out",type=int, default=30, help='max length of output')
parser.add_argument("--similarity",type=str, default='cosine', help='similarity measure to use')
parser.add_argument("--encoder",type=str, default='lstm', help='encoder type to use')

args = parser.parse_args()

def train(args):
    print(args)
    data_loader = get_loader(args.train_root, args.ptr, args.batch, args.max_out)
    if args.ptr:
        from models.pointer_network import PtrNet
        model = PtrNet(vocab_size=args.vocab, embed_size=args.embed, hidden_size=args.hidden,
                       num_layers=args.n_layers,max_out=args.max_out)
        criterion = nn.NLLLoss()
    else:
        from models.bigru_attn import biGRU_attn
        model = biGRU_attn(vocab_size=args.vocab, embed_size=args.embed, hidden_size=args.hidden,
                     num_layers=args.n_layers)
        criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda()
    steps = 0
    opt = optim.Adam(model.parameters(), lr=args.lr)
    total_batches=0
    for epoch in range(args.epochs):
        within_steps = 0
        for i, (sources, targets, labels) in enumerate(data_loader):
            # split tuples
            steps+=1
            if steps==100000:
                sys.exit()
            total_batches = max(total_batches,i)
            model.zero_grad()
            if args.cuda:
                sources = sources.cuda()
                targets = targets.cuda()
                labels = labels.cuda()
            sources = Variable(sources)
            targets = Variable(targets)
            labels = Variable(labels)
            outputs = model.train(sources,targets)
            if args.ptr:
                packed_outputs, packed_targets = pack_padded(outputs, labels)
            else:
                packed_outputs, packed_targets = pack_padded(outputs, targets[:,1:])
            loss = criterion(packed_outputs,packed_targets)
            loss.backward()
            print(loss.data[0])
            # free memory
            del targets, outputs, packed_targets, packed_outputs
            opt.step()
            if steps%10==0:
                print('validation------------------------')
                val(args,model)
                print('end------------------------')

def val(args,model):
    data_loader = get_loader(args.test_root, args.ptr, 100, args.max_out)
    if args.ptr:
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    steps = 0
    opt = optim.Adam(model.parameters(), lr=args.lr)
    total_batches=0
    within_steps = 0
    for i, (sources, targets, labels) in enumerate(data_loader):
        # split tuples
        if args.cuda:
            sources = sources.cuda()
            targets = targets.cuda()
            labels = labels.cuda()
        sources = Variable(sources)
        targets = Variable(targets)
        labels = Variable(labels)
        outputs = model.test(sources)
        if args.ptr:
            packed_outputs, packed_targets = pack_padded(outputs, labels)
        else:
            packed_outputs, packed_targets = pack_padded(outputs, targets[:,1:])
        loss = criterion(packed_outputs,packed_targets)
        correct = (packed_outputs.max(1)[1].data==packed_targets.data).long().sum()
        print(correct*1.0/len(packed_targets))
        print(loss.data[0])
        # free memory
        del targets, outputs, packed_targets, packed_outputs
            
def main(args):
    if args.mode=='train':
        print("Train mode")
        train(args)
    elif args.mode=='test':
        print("Test mode")
        test(args)
    else:
        print("Error: please specify --mode as 'train' or 'test'")
        
if __name__ == "__main__":
    main(args)