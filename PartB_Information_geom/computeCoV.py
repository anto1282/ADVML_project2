import numpy as np
import matplotlib.pyplot as plt


# Load data and compute distances
geodesiclength = np.zeros([10, 25, 3])
euclidianlength = np.zeros([10, 25, 3])

for i in range(3):
    for j in range(10): 
        data = np.load(f"geodesic_data/model_run{j}_dec{i+1}_data.npz", allow_pickle=True)
        k = 0
        for val in data["losses"]:
            geodesiclength[j, k, i] = np.sqrt(val)
            k += 1 
        euclidianlength[j, :, i] = np.linalg.norm(data["end_points"] - data["start_points"], axis=1)

# Compute CoV (standard deviation divided by mean) for each point pair across 10 runs
CoV_geodesic_enc1 = np.std(geodesiclength[:, :, 0], axis=0) / np.mean(geodesiclength[:, :, 0], axis=0)
CoV_geodesic_enc2 = np.std(geodesiclength[:, :, 1], axis=0) / np.mean(geodesiclength[:, :, 1], axis=0)
CoV_geodesic_enc3 = np.std(geodesiclength[:, :, 2], axis=0) / np.mean(geodesiclength[:, :, 2], axis=0)

CoV_euclidian_enc1 = np.std(euclidianlength[:, :, 0], axis=0) / np.mean(euclidianlength[:, :, 0], axis=0)
CoV_euclidian_enc2 = np.std(euclidianlength[:, :, 1], axis=0) / np.mean(euclidianlength[:, :, 1], axis=0)
CoV_euclidian_enc3 = np.std(euclidianlength[:, :, 2], axis=0) / np.mean(euclidianlength[:, :, 2], axis=0)

# Organize the CoV values for plotting
x_vals = [1, 2, 3]
CoV_geodesic = [CoV_geodesic_enc1, CoV_geodesic_enc2, CoV_geodesic_enc3]
CoV_euclidian = [CoV_euclidian_enc1, CoV_euclidian_enc2, CoV_euclidian_enc3]

# Compute the overall mean CoV for each decoder (averaged across the 25 point pairs)
mean_CoV_geodesic = [np.mean(CoV_geodesic_enc1), np.mean(CoV_geodesic_enc2), np.mean(CoV_geodesic_enc3)]
mean_CoV_euclidean = [np.mean(CoV_euclidian_enc1), np.mean(CoV_euclidian_enc2), np.mean(CoV_euclidian_enc3)]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define marker style and colors for individual points
colors = ['r', 'g', 'b']

# Plot geodesic CoV values
for i in range(3):
    ax1.scatter([x_vals[i]] * 25, CoV_geodesic[i],
                s=70,color=colors[i],
                label=f'{i+1} decoders', alpha = 0.5)  # Only label first instance to avoid duplicates

# Overlay the mean values as diamonds and connect with a dashed line
ax1.scatter(x_vals, mean_CoV_geodesic, s=150, marker='o', color='black', label='Mean CoV')
ax1.plot(x_vals, mean_CoV_geodesic, color='black', linestyle='-', alpha = 0.7)

ax1.set_title("CoV of Geodesic Distances", fontsize=14)
ax1.set_xlabel("Number of Decoders", fontsize=12)
ax1.set_ylabel("Coefficient of Variation", fontsize=12)
ax1.set_xticks(x_vals)
ax1.legend(fontsize=11)
ax1.grid(True)

# Plot Euclidean CoV values
for i in range(3):
    ax2.scatter([x_vals[i]] * 25, CoV_euclidian[i],
                s=70, color=colors[i],
                label=f'{i+1} decoders', alpha = 0.5)
    
# Overlay the mean values as diamonds and connect with a dashed line
ax2.scatter(x_vals, mean_CoV_euclidean, s=150, marker='o', color='black', label='Mean CoV')
ax2.plot(x_vals, mean_CoV_euclidean, color='black', linestyle='-', alpha = 0.7)

ax2.set_title("CoV of Euclidean Distances", fontsize=14)
ax2.set_xlabel("Number of Decoders", fontsize=12)
ax2.set_xticks(x_vals)
ax2.legend(fontsize=11)
ax2.grid(True)

plt.tight_layout()
plt.savefig("CoV_with_mean.png")
plt.show()
