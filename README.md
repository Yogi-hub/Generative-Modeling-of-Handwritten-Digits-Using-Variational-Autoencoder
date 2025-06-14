# Variational Auto-encoder

You’ll find all relevant code in:

- `vae.py` – Model architecture and training logic  
- `task1.ipynb` – Training script, evaluation, visualization, and sampling  
- `best_model.pt` – Best model weights

## Download and prepare dataset
```python
# Get current working directory
DATA_DIR = os.path.join(os.getcwd(), 'datasets', 'mnist')

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0, 1] range
])

dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform)

# Split training into training and validation
train_dataset, val_dataset = random_split(dataset, [50000, 10000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128);
```

## To import model and utilities

```python
from vae import VAE, train_vae, vae_loss_gaussian
```

## Load pretrained model

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=20).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
model.eval()
