import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

import models

def compute_loss(x, x_recon, z, v_dist, model, args):
    if args.kld == 'eric':
        bce, kl = model.approximate_loss(x, x_recon, v_dist)
    else:
        bce, kl = model.loss(x, x_recon, z, v_dist)
    elbo = -bce - kl
    loss = -elbo
    return loss

def train(epoch, model, train_loader, optimizer, device, args):
    model = model.train()

    train_loss = 0.
    for i, (x, _) in enumerate(train_loader, 1):
        x = x.to(device)
        x = x.view(x.size(0), -1)

        # scheduler for temperature
        if i % args.temp_interval == 0:
            n_updates = epoch * len(train_loader) + i
            temp = max(torch.tensor(args.init_temp) * np.exp(-n_updates*args.temp_anneal), torch.tensor(args.min_temp))
        
        # compute & optimize the loss function
        z, x_recon, v_dist = model(x)
        loss = compute_loss(x, x_recon, z, v_dist, model, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()

    # report results
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

def test(epoch, model, test_loader, device, args):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(x.size(0), -1)

            # compute the loss function
            z, x_recon, v_dist = model(x)
            loss = compute_loss(x, x_recon, z, v_dist, model, args)
            test_loss += loss

            # save reconstructed figure
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n].reshape(-1, 1, 28, 28),
                                        x_recon.view(args.batch_size, 1, 28, 28)[:n]])
                if 'results' not in os.listdir():
                    os.mkdir('results')
                save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    # report results
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 4} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    Model = getattr(models, args.sampling)
    model = Model(args.init_temp, args.latent_num, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, device, args)
        test(epoch, model, test_loader, device, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='.data/',
                        help='where is you mnist?')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--latent-num', type=int, default=20, metavar='N',
                        help='the number of latent variables')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                        help='the dimension for each latent variables')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--init-temp', type=float, default=1.0)
    parser.add_argument('--temp-anneal', type=float, default=0.00009)
    parser.add_argument('--temp-interval', type=float, default=300)
    parser.add_argument('--min-temp', type=float, default=0.5)

    parser.add_argument('--sampling', type=str, default='TDModel',
                        help='example: TDModel utilizes torch.distributions.relaxed, ExpTDModel stabilizes loss function')
    parser.add_argument('--kld', type=str, default='eric',
                        help='example: eric, madisson')

    args = parser.parse_args()
    main(args)