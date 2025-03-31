import numpy as np

data = np.load("geodesic_data/model_run0_dec3_data.npz", allow_pickle=True)

# Access the saved arrays
latents = data["latent_points"]
labels = data["labels"]
geodesics = data["geodesic_points"]  # This will be an array of objects (lists/arrays), hence allow_pickle=True
start_points = data["start_points"]
end_points = data["end_points"]
start_classes = data["start_classes"]
end_classes = data["end_classes"]
losses = data["losses"]


print(losses)
print(geodesics)
