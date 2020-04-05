import torch
from torch import nn, optim

class Net(nn.Module):
    def __init__(self, latent_size):
        super(Net, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True), 

            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True), 
        )
        
        self.mu = nn.Linear(64, latent_size)
        self.logvar = nn.Linear(64, latent_size)
        self.relu = nn.ReLU(True)
        
        # self.remap = nn.Linear(latent_size, 32)
        # self.remap = nn.Sequential(
        #     nn.Linear(latent_size, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True), 
        # )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True), 

            nn.Linear(64, 28*28),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, input_img):
        input_flat = input_img.view(-1, 28*28)
        x = self.encoder(input_flat)
        mu = self.relu(self.mu(x))
        logvar = self.relu(self.logvar(x))
        sample = self.reparameterize(mu, logvar)

        return sample, mu, logvar      

    def decode(self, sample):
        # remap = self.remap(sample)
        result = self.decoder(sample)
        return result   

    def forward(self, input_img):
        sample, mu, logvar = self.encode(input_img) 
        result = self.decode(sample)
        
        return result, mu, logvar