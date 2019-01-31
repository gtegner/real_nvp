import argparse
import torch
import torchvision
from torchvision.transforms import transforms
from torch import distributions
import matplotlib.pyplot as plt
import os
import numpy as np


from model import CNN_Net
def train(net, train_loader):
    
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr = 1e-3)

    num_epochs = 100
    for epoch in range(num_epochs):
        for i, (img, _) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -net.log_prob(img).mean()
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print('Epoch: {} - Loss: {}'.format(epoch, loss))


def plot_sample(net):

    samples = net.sample(5)
    for i in range(len(s)):
        plt.imshow(samples.detach().numpy()[i,0,:,:])
        plt.show()
        
def main():
    if not os.path.exists('data'):
        os.mkdir('data')

    batch_size = 8
    trainset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    prior = distributions.MultivariateNormal(torch.zeros((28,28)), torch.eye(28))
    masks = torch.from_numpy(np.array([np.resize([1,0], [28,28]),np.resize([0,1], [28,28])]*2).astype(np.float32))
    net = CNN_Net(masks, prior)

    train(net, train_loader)

if __name__ == '__main__':
    main()
