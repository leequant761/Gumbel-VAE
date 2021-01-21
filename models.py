# Main Reference : https://github.com/daandouwe/concrete-vae
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

from modules import Encoder, Decoder

class Model(nn.Module):
    def __init__(self, temp, latent_num, latent_dim):
        super(Model, self).__init__()
        if type(temp) != torch.Tensor:
            temp = torch.tensor(temp)
        self.temp = temp
        self.latent_num = latent_num
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_num=latent_num, latent_dim=latent_dim)
        self.decoder = Decoder(latent_num=latent_num, latent_dim=latent_dim)
        self.prior = dist.RelaxedOneHotCategorical(self.temp, probs=torch.ones(latent_dim).cuda())
        self.initialize()

        self.softmax = nn.Softmax(dim=-1)

    def initialize(self):
        for param in self.parameters():
            if len(param.shape) > 2:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        log_alpha = self.encode(x)
        z = self.sample(log_alpha, self.temp)
        x_recon = self.decode(z.view(-1, self.latent_num*self.latent_dim))
        return log_alpha, z, x_recon

    def approximate_loss(self, x, x_recon, log_alpha):
        """ KL-divergence follows Eric Jang's trick
        """
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        num_class = torch.tensor(self.latent_dim).float()
        probs = torch.softmax(log_alpha, dim=-1) # alpha_i / alpha_sum
        kl = (probs * (num_class * probs).log()).sum()
        return bce, kl

    def loss(self, x, x_recon, z, log_alpha):
        """ Monte-Carlo estimate KL-divergence
        """
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        n_batch = x.shape[0]
        prior = self.prior.expand(torch.Size([n_batch, self.latent_num]))
        v_dist = dist.RelaxedOneHotCategorical(self.temp, logits=log_alpha)
        kl = (v_dist.log_prob(z) - prior.log_prob(z)).sum()
        return bce, kl

    def sample(self, log_alpha, temp):
        raise ValueError("Not Implemented")

class NaiveModel(Model):
    def sample(self, log_alpha, temp):
        gumbel = dist.Gumbel(torch.zeros_like(log_alpha), torch.ones_like(log_alpha)).sample()
        return torch.softmax((log_alpha + gumbel) / temp, dim=-1)

class TDModel(Model):
    def sample(self, log_alpha, temp):
        concrete = dist.RelaxedOneHotCategorical(temp, logits=log_alpha).rsample()
        return concrete