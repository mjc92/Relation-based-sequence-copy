import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# RNN Based Language Model
class RelPtrNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, max_out):
        super(RelPtrNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(hidden_size*2, hidden_size, num_layers, 
                                batch_first=True)
        self.W1 = nn.Linear(hidden_size*2, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size*2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.hidden_size = hidden_size
        
        self.max_out = max_out
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        
    def train(self, enc_in, dec_in, teacher_forcing=True):
        # enc_in : [b x in_seq]
        # dec_in : [b x out_seq]
        
        # Encoder
        enc_embedded = self.embed(enc_in)
        encoded, _ = self.encoder(enc_embedded) # [b x in_seq x hidden*2]
        
        # get last states of encoder
        sizes = (enc_in>0).long().sum(1).data
        last_states = torch.stack([encoded[i,x-1] for i,x in enumerate(sizes)],0) # [b x hidden*2]
        
        # Decoder
        state = self.W1(last_states)
        outputs = []
        
        for i in range(dec_in.size(1)+1):
            if i==0:
                context = Variable(torch.FloatTensor(dec_in.size(0),
                    1,self.hidden_size*2).zero_())#.cuda()# get initial context, [b x 1 x h*2]
            # if teacher_forcing==True:
            #     input = context
            #     if i==0:
            #         input = dec_embedded[:,0].unsqueeze(1)
            #     else:
            #         next_words = self.linear(out.squeeze())
            #         next_idx = next_words.max(1)[1]
            #         input = self.embed(next_idx).unsqueeze(1)
            #     input = torch.cat([context,input],dim=2)
            
            out, state = self.decoder(context, state.unsqueeze(0))
            state = state.squeeze()
            comp = self.W2(state) # [batch x hidden*2]
            scores = torch.bmm(encoded,comp.view(comp.size(0),-1,1)) # [b x seq x 1]
            scores = F.softmax(scores.squeeze())
            """
            [b x 1 x h*2] = [b x 1 x seq] x [b x seq x h*2]
            """
            context = torch.bmm(scores.view(scores.size(0),1,-1),encoded) # [b x 1 x h*2]
            outputs.append(scores)
        outputs = torch.stack(outputs,dim=1) # [b x seq (seq_length) x seq (output_dim)]
        diff = self.max_out-outputs.size(2)
        if diff>0:
            tmp = Variable(torch.zeros(outputs.size(0),outputs.size(1),diff))
        outputs = torch.cat([outputs,tmp],2)
        
        return torch.log(outputs+1e-5)

    def test(self, enc_in, teacher_forcing=True):
        # enc_in : [b x in_seq]
        
        # Encoder
        enc_embedded = self.embed(enc_in)
        encoded, _ = self.encoder(enc_embedded) # [b x in_seq x hidden*2]
        
        # get last states of encoder
        sizes = (enc_in>0).long().sum(1).data
        last_states = torch.stack([encoded[i,x-1] for i,x in enumerate(sizes)],0) # [b x hidden*2]
        
        # Decoder
        state = self.W1(last_states)
        outputs = []
        
        for i in range(self.max_out):
            if i==0:
                context = Variable(torch.FloatTensor(encoded.size(0),
                    1,self.hidden_size*2).zero_())#.cuda()# get initial context, [b x 1 x h*2]
            out, state = self.decoder(context, state.unsqueeze(0))
            state = state.squeeze()
            comp = self.W2(state) # [batch x hidden*2]
            scores = torch.bmm(encoded,comp.view(comp.size(0),-1,1)) # [b x seq x 1]
            scores = F.softmax(scores.squeeze())
            """
            [b x 1 x h*2] = [b x 1 x seq] x [b x seq x h*2]
            """
            context = torch.bmm(scores.view(scores.size(0),1,-1),encoded) # [b x 1 x h*2]
            outputs.append(scores)
        outputs = torch.stack(outputs,dim=1) # [b x seq (seq_length) x seq (output_dim)]
        diff = self.max_out-outputs.size(2)
        if diff>0:
            tmp = Variable(torch.zeros(outputs.size(0),outputs.size(1),diff))
        outputs = torch.cat([outputs,tmp],2)
        
        return torch.log(outputs+1e-5)