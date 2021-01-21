# Main Reference : https://github.com/daandouwe/concrete-vae
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

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
        if 'ExpTDModel' in  str(self.__class__):
            self.prior = ExpRelaxedCategorical(self.temp, probs=torch.ones(latent_dim).cuda())
        else:
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
        z, v_dist = self.sample(log_alpha, self.temp)
        if 'ExpTDModel' in  str(self.__class__):
            x_recon = self.decode(z.exp().view(-1, self.latent_num*self.latent_dim))
        else:
            x_recon = self.decode(z.view(-1, self.latent_num*self.latent_dim))
        return z, x_recon, v_dist

    def approximate_loss(self, x, x_recon, v_dist):
        """ KL-divergence follows Eric Jang's trick
        """
        log_alpha = v_dist.logits
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        num_class = torch.tensor(self.latent_dim).float()
        probs = torch.softmax(log_alpha, dim=-1) # alpha_i / alpha_sum
        kl = (probs * (num_class * probs).log()).sum()
        return bce, kl

    def loss(self, x, x_recon, z, v_dist):
        """ Monte-Carlo estimate KL-divergence
        """
        bce = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        n_batch = x.shape[0]
        prior = self.prior.expand(torch.Size([n_batch, self.latent_num]))
        kl = (v_dist.log_prob(z) - prior.log_prob(z)).sum()
        return bce, kl

    def sample(self, log_alpha, temp):
        raise ValueError("Not Implemented")

class TDModel(Model):
    def sample(self, log_alpha, temp):
        v_dist = dist.RelaxedOneHotCategorical(temp, logits=log_alpha)
        concrete = v_dist.rsample()
        return concrete, v_dist

class ExpTDModel(Model):
    def sample(self, log_alpha, temp):
        v_dist = ExpRelaxedCategorical(temp, logits=log_alpha)
        log_concrete = v_dist.rsample()
        return log_concrete, v_dist