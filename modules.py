import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_num=30, latent_dim=10, hidden=200, output=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_num*latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.act_fn = nn.Tanh()

    def forward(self, z):
        h = self.fc1(z)
        h = self.act_fn(h)
        return self.fc2(h)


class Encoder(nn.Module):
    def __init__(self, input=784, hidden=200, latent_num=30, latent_dim=10):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, latent_num * latent_dim)
        self.act_fn = nn.Tanh()
        self.latent_num = latent_num
        self.latent_dim = latent_dim

    def forward(self, x):
        h = self.fc1(x)
        h = self.act_fn(h)
        return self.fc2(h).reshape(-1, self.latent_num, self.latent_dim)