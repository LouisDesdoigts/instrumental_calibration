import jax.numpy as np
import paths
import dill as p
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

# Load model
# tel = p.load(open(paths.data / 'instrument.p', 'rb'))
# model = p.load(open(paths.data / 'model.p', 'rb'))
# source = p.load(open(paths.data / 'source.p', 'rb'))
data = np.load(paths.data / "make_model_and_data/data.npy")
final_psfs = np.load(paths.data / "optimise/final_psfs.npy")
initital_psfs = np.load(paths.data / "make_model_and_data/initial_psfs.npy")

plt.figure(figsize=(15, 4.5))
plt.suptitle("Data and Residuals", size=15)

ax1 = plt.subplot(1, 3, 1)
plt.title("Sample Data")
plt.imshow(data*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")

ax2 = plt.subplot(1, 3, 2)
plt.title("Initial Residual")
plt.imshow((initital_psfs - data)*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")

ax3 = plt.subplot(1, 3, 3)
plt.title("Final Residual")
plt.imshow((final_psfs - data)*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")

# inset axes....
axins = ax3.inset_axes([0.5, 0.5, 0.45, 0.45])
axins.imshow((final_psfs - data)*1e-3)
# sub region of the original image
axins.set_xlim(400, 430)
axins.set_ylim(400-30, 400)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax3.indicate_inset_zoom(axins, edgecolor="black")

plt.tight_layout()
plt.savefig(paths.figures / "data_resid.pdf", dpi=300)
