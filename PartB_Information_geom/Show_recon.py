import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from ensemble_vae import VAE,GaussianDecoder,GaussianEncoder,GaussianPrior
from torch import nn
M = 2
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild the model
model = VAE(
    GaussianPrior(M),
    GaussianDecoder([new_decoder().to(device) for _ in range(3)]),
    GaussianEncoder(new_encoder()),
).to(device)

model_name = "model_run0_dec3"
model.load_state_dict(torch.load(f"experiment/{model_name}.pt", map_location=device))
model.eval()

# Load geodesic data
data = np.load(f"geodesic_data/{model_name}_data.npz", allow_pickle=True)
geodesics = data["geodesic_points"]
z_path = torch.tensor(geodesics[0], dtype=torch.float32).to(device)  # pick first path

# Decode with all 3 decoders
decoded_imgs = []
with torch.no_grad():
    for decoder in model.decoder.decoder_nets:
        recon = decoder(z_path).cpu()  # (T, 1, 28, 28)
        decoded_imgs.append(recon)

# --- Plot ---
n_steps = z_path.shape[0]
fig, axes = plt.subplots(nrows=3, ncols=n_steps, figsize=(n_steps, 3*3))

for row in range(3):
    for col in range(n_steps):
        ax = axes[row, col]
        ax.imshow(decoded_imgs[row][col][0], cmap='gray')
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(f'Decoder {row}', fontsize=12)

plt.suptitle("Geodesic Reconstructions from All Decoders", fontsize=16)
plt.tight_layout()
plt.savefig("Recon.png")
