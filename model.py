import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.parameter import Parameter



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def forward(self, input):
        return input.view(-1,*(1,28,28))

def cnn_size(W,F,P,S):
    return (W-F+2 * P)/S + 1
    
class CNN_Net(nn.Module):
    def __init__(self, masks, prior):
        super(CNN_Net, self).__init__()

   
        self.prior = prior
        self.masks = masks
          
        n_masks = masks.size(0)
        
        
        self.t = nn.ModuleList([nn.Sequential(nn.Conv2d(1,4,3), nn.LeakyReLU(), 
                              nn.Conv2d(4,8,3), nn.LeakyReLU(),nn.MaxPool2d(kernel_size = 2),
                                              Flatten(),
                                              nn.Linear(1152,28*28), 
                                              nn.Tanh(), Reshape()) for i in range(n_masks)])
        
        self.s = nn.ModuleList([nn.Sequential(nn.Conv2d(1,4,3), nn.LeakyReLU(), 
                              nn.Conv2d(4,8,3), nn.LeakyReLU(), nn.MaxPool2d(kernel_size = 2),
                                              Flatten(), 
                                              nn.Linear(1152,28*28), 
                                              nn.Tanh(), Reshape()) for i in range(n_masks)])
        
                
        
      
    # x->z
    def forward(self, x):
        x_ = x
        log_det = 0
        for i in range(len(self.t)):
            y1 = self.masks[i] * x_
            y2 = (1-self.masks[i])* (x_ *torch.exp(self.s[i](y1)) + self.t[i](y1))

            x_ = y1 + y2
            log_det += self.s[i](y1).sum(dim = 1)

        return x_, log_det

    #Generating  z -> x
    def backward(self, z):
        y = z
        for i in reversed(range(len(self.t))):
            x1 = self.masks[i] * y
            x2 = ((1-self.masks[i]) * y - self.t[i](x1)) * torch.exp(-self.s[i](x1))
            y = x1 + x2

        return y

    def log_prob(self, x):
        z, log_det = self.forward(x)

        return self.prior.log_prob(z) + log_det
    
    def sample(self, batch_size):
        z = self.prior.sample((batch_size,1))
        x = self.backward(z)
        return x
        