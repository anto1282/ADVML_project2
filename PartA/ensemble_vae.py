# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net
    def calc_c_dot(c):
        c_shifted_fwd = torch.cat((c[1:], c[-1:]), dim=0)
        c_shifted_back = torch.cat((c[:1], c[:-1]), dim=0)
        return (c_shifted_fwd - c_shifted_back) / 2
    
    def eval_J(model,latent_x):
        latent_x = latent_x.unsqueeze(0)
        jacobian = torch.autograd.functional.jacobian(model.decoder.decoder_net.forward, latent_x, 
                                                    create_graph=False, strict=False, 
                                                    vectorize=True, strategy='reverse-mode')
            
        J = jacobian.squeeze(0).squeeze(0).squeeze(2).flatten(0,1)
        return J


    def plot_path(x_inner, x0, xN, step,title="Path", show_points=True, show_line= True, direc= "Optimizations"):
        """
        Visualize the full path [x0, x_inner..., xN] in 2D.

        Args:
            x_inner: torch.Tensor of shape (N, 2)
            x0: torch.Tensor of shape (2,)
            xN: torch.Tensor of shape (2,)
            title: Title of the plot
            show_points: If True, annotate start and end
        """
        with torch.no_grad():
            x_full = torch.cat([x0, x_inner, xN], dim=0)

            xs = x_full[:, 0].cpu().numpy()
            ys = x_full[:, 1].cpu().numpy()

            plt.figure(figsize=(6, 6))
            plt.plot(xs, ys, '-o' if show_line else "o", label='Path')
            if show_points:
                plt.text(xs[0], ys[0], 'Start', fontsize=10, color='green')
                plt.text(xs[-1], ys[-1], 'End', fontsize=10, color='red')

            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(title)
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
            plt.savefig(f"{direc}/{title}_at_step_{step}.png")

    def save_geodesic_data_npz(latents, labels, geodesics, pairs, pair_labels, path="geodesics_data.npz"):
            np_geodesics = [g.cpu().detach().numpy() for g in geodesics]
            np_starts = [p[0].cpu().detach().numpy() for p in pairs]
            np_ends = [p[1].cpu().detach().numpy() for p in pairs]
            np_classes = np.array(pair_labels)

            np.savez(path,
                    latent_points=latents.cpu().numpy(),
                    labels=labels.cpu().numpy(),
                    geodesic_points=np_geodesics,
                    start_points=np_starts,
                    end_points=np_ends,
                    start_classes=np_classes[:, 0],
                    end_classes=np_classes[:, 1])
    
    def plot_latent_space_with_geodesics(latents, labels, geodesics, pairs, save_path="geodesic_plot.png"):
        latents = latents.cpu().numpy()
        labels = labels.cpu().numpy()

        plt.figure(figsize=(10, 10))

        # Define three colors for the three classes
        color_map = {0: 'orange', 1: 'skyblue', 2: 'purple'}

        for cls in [0, 1, 2]:
            mask = labels == cls
            plt.scatter(
                latents[mask, 0], latents[mask, 1],
                alpha=0.3, label=f"Class {cls}", color=color_map[cls]
            )

        # Plot geodesics
        for i,geo in enumerate(geodesics):
            geo = geo.detach().cpu().numpy()
            plt.plot(geo[:, 0], geo[:, 1], linewidth=2, color='yellow', label = "Geodesic" if i == 0 else None)
            
            start = geo[0]
            end = geo[-1]

            # Connect endpoints with dashed line
            plt.plot([start[0], end[0]], [start[1], end[1]], linestyle='dashed', color='blue', alpha=0.5, label="Straight line" if i == 0 else None)

            # Highlight start and end
            plt.scatter([start[0]], [start[1]], color='green', s=60, marker='o')
            plt.scatter([end[0]], [end[1]], color='green', s=60, marker='o')

        plt.title("Latent Space with Geodesics by Class")
        plt.axis('equal')
        plt.legend()
        plt.savefig(save_path)
    

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()
        all_labels = []
        all_latents = []
        
        with torch.no_grad():
            for x, labels in mnist_test_loader:
                x = x.to(device)
                z = model.encoder(x).mean  # or .rsample() if you want stochastic encoding
                all_latents.append(z)
                all_labels.append(labels)

        all_latents = torch.cat(all_latents, dim=0)  # shape: (N, latent_dim)
        all_labels = torch.cat(all_labels,dim=0)

        import json

        n_pairs = 25
        pair_file = "geodesic_pair_indices.json"

        if os.path.exists(pair_file):
            print(f"Loading geodesic pairs from {pair_file}")
            with open(pair_file, "r") as f:
                pair_indices = json.load(f)
        else:
            print(f"{pair_file} not found. Generating new pairs and saving...")
            N = all_latents.size(0)
            all_indices = torch.randperm(N)[:2 * n_pairs]
            pair_indices = [(all_indices[i].item(), all_indices[i + 1].item()) for i in range(0, 2 * n_pairs, 2)]

            with open(pair_file, "w") as f:
                json.dump(pair_indices, f, indent=2)

        pairs = [(all_latents[i], all_latents[j]) for i, j in pair_indices]
        pair_labels = [(all_labels[i], all_labels[j]) for i, j in pair_indices]

        from pathlib import Path

        geodesics = []

        for i,(x0,xN) in enumerate(pairs):

            x0 = x0.unsqueeze(0)
            xN = xN.unsqueeze(0)
            
            dims = 2
            num_steps = 20

            x_inner = torch.linspace(0, 1, num_steps).unsqueeze(1).to(device) * (xN - x0) + x0
            x_inner = torch.nn.Parameter(x_inner,requires_grad = True).to(device)
            optimizer = torch.optim.Adam([x_inner], lr=0.01)

            opti_dir = f"Optimizations/Pair_{i}_optimization graphs"
            Path(opti_dir).mkdir(parents=True, exist_ok=True)

            loss_clone = 0
            loss = 0
            for step in tqdm(range(50),desc=loss):
                x = torch.cat([x0, x_inner, xN], dim=0).to(device)
                if step == 0:
                    x_start = x_inner
                c_dot = calc_c_dot(x)
                                
                loss = 0.0
                S = len(c_dot)
                for i, c_dot_s in enumerate(c_dot):
                    #The jacobian at point x[i]
                    J = eval_J(model, x[i])

                    #Loss == energy
                    loss += (c_dot_s.T @J.T @ J @ c_dot_s) / S

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()

                if step % 25 == 0:
                    print(f"Step {step}: loss = {loss.item():.4f}")
                    plot_path(x_inner, x0, xN,step, title=f"During Optimization at step {step+1}", direc=opti_dir)
            x0 = x0.unsqueeze(0) if x0.dim() == 1 else x0
            xN = xN.unsqueeze(0) if xN.dim() == 1 else xN
            x_inner = x_inner if x_inner.dim() == 2 else x_inner.unsqueeze(0)  # Ensure 2D

            geodesics.append(torch.cat([x0, x_inner.detach(), xN], dim=0))

        pair_labels = []
        for start, end in pairs:
            start_idx = torch.argmin(torch.norm(all_latents - start.unsqueeze(0), dim=1))
            end_idx = torch.argmin(torch.norm(all_latents - end.unsqueeze(0), dim=1))
            pair_labels.append((all_labels[start_idx].item(), all_labels[end_idx].item()))

        
            
        save_geodesic_data_npz(all_latents, all_labels, geodesics, pairs, pair_labels, path="geodesics_data.npz")
        plot_latent_space_with_geodesics(all_latents, all_labels, geodesics, pairs)

            
            


