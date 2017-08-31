import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(input_size,hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Forward propagate RNN
        x = self.embed(x)
        encoded, _ = self.encoded(x) # [b x seq x hidden]
        
        
        return out # [b x seq x hidden]