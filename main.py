import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from vae import VAE, train_vae, reparameterize 
from sklearn.decomposition import PCA

def main():

    def latent_space_2D(samples, labels):

        # Instantiate and load the model
        latent_dim = 2
        model = VAE(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load("latent_space_2d.pt", map_location=device, weights_only=True))

        # Set the model to evaluation mode
        model.eval()

        with torch.no_grad():
            mu, logvar = model.encoder(samples)
            z = reparameterize(mu, logvar).cpu().numpy()

        plt.figure()

        x = z[:,0]
        y = z[:,1]

        scatter_plot = plt.scatter(x, y, c=labels, cmap="plasma", s= 10)
        plt.colorbar(scatter_plot, ticks=range(10), label='Associated colour for labels')
        plt.title("Latent space in 2 Dimension")
        plt.show()

    def PCA_2D(samples , labels):

        # Instantiate and load the model
        model = VAE(latent_dim=15).to(device)
        model.load_state_dict(torch.load("latent_space_15.pt", map_location=device, weights_only=True))

        # Set the model to evaluation mode
        model.eval()

        with torch.no_grad():
            mu, logvar = model.encoder(samples)
            z = reparameterize(mu, logvar).cpu().numpy()

        pca = PCA(n_components=2, random_state = 1000)
        main_components = pca.fit_transform(z)

        plt.figure()

        x = main_components[:,0]
        y = main_components[:,1]

        scatter_plot = plt.scatter(x, y, c=labels, cmap="plasma", s=10)
        plt.colorbar(scatter_plot, ticks=range(10), label='Associated colour for labels')
        plt.title("Latent space after PCA")
        plt.show()

    def linear_interpolation(test_loader):

        # Instantiate and load the model
        model = VAE(latent_dim=20).to(device)
        model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True)) # Uses Gaussian model

        # Set the model to evaluation mode
        model.eval()
        k = 5

        fig, axes = plt.subplots(5, k+2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.2 , hspace=0.6)

        images, labels = next(iter(test_loader))

        for row, i in enumerate(range(0, 9, 2)):

            image_1 = images[i].to(device)
            label_1 = labels[i].to(device)

            image_2 = images[i+1].to(device)
            label_2 = labels[i+1].to(device)

            with torch.no_grad():
                mu1, logvar1 = model.encoder(image_1.unsqueeze(0))
                z1 = reparameterize(mu1, logvar1)

                mu2, logvar2 = model.encoder(image_2.unsqueeze(0))
                z2 = reparameterize(mu2, logvar2)

                lambda_k = np.linspace(0,1, k+2)
                z_linear_interpolated = [(lambda_1 * z1 + (1 - lambda_1) * z2) for lambda_1 in lambda_k]
                interpolated_image = [model.decoder(z) for z in z_linear_interpolated]

            for idx, img in enumerate(interpolated_image):
                ax = axes[row , idx]
                ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
                ax.axis('off')

            axes[row,0].set_title(f"Label : {label_2}")
            axes[row,k+1].set_title(f"Label : {label_1}")

        plt.suptitle("Linear Interpolation in the Latent space")
        plt.tight_layout()
        plt.show()

    if __name__ == '__main__':

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        DATA_DIR = os.path.join(os.getcwd(), 'datasets', 'mnist')

        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

        test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000)

        #TASK 3
        samples , labels = next(iter(test_loader))
        samples = samples.to(device)
        labels = np.array(labels)

        # TASK 3A
        latent_space_2D(samples, labels)

        #TASK 3B
        PCA_2D(samples, labels)

        #TASK 3C
        linear_interpolation(test_loader)

main()