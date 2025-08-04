import torch
import numpy as np


' The folowing class is what I use to make all my NN'

class Net(torch.nn.Module):
    def __init__(self, act, input_size,  *args):
        super(Net, self).__init__()
        self.act = act # this is out activation function, it can be torch.relu or torch.cos, or anything you want to make
        self.hidden_layers = torch.nn.ModuleList()
        for layer_size in args:
            self.hidden_layers.append(torch.nn.Linear(input_size, layer_size))
            input_size = layer_size

        # Output layer
        self.output_layer = torch.nn.Linear(input_size, 1)
        self.initialize_weights
    
    def initialize_weights(self):
        for layer in self.hidden_layers:
            if self.act == torch.relu:
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else:
                torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.1)
        
        if self.act == torch.relu:
            torch.nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        else:
            torch.nn.init.xavier_normal_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.output_layer(x)
        return x

        
"""Below is a function used to evaluate an optimized nueral network
and make the outputs ready to be plotted."""

def net_eval(net, inputs):
    out = net(inputs)
    return out.detach()