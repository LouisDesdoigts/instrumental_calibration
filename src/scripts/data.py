# Core jax
import jax
import jax.numpy as np
import jax.random as jr

# Optimisation
import equinox as eqx
import optax

# Optics
import dLux as dl
from dLux.utils import arcseconds_to_radians as a2r
from dLux.utils import radians_to_arcseconds as r2a

# Paths
import paths

# Pickle
# import pickle as p
import dill as p

# Plotting/visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Load model
tel = p.load(open(paths.data / 'instrument.p', 'rb'))
model = p.load(open(paths.data / 'model.p', 'rb'))
source = p.load(open(paths.data / 'source.p', 'rb'))
data = np.load(paths.data / "data.npy")

plt.figure(figsize=(15, 4.5))
plt.suptitle("Data and Residuals", size=15)

ax1 = plt.subplot(1, 3, 1)
plt.title("Sample Data")
plt.imshow(data[0]*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")


initital_psfs = model.observe()
ax2 = plt.subplot(1, 3, 2)
plt.title("Initial Residual")
plt.imshow((initital_psfs[0] - data[0])*1e-3)
# plt.imshow(data[0]/initital_psfs[0])
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")


final_model = p.load(open(paths.data / 'models_out.p', 'rb'))[-1]
final_psfs = final_model.observe()

# ax3 = plt.subplot(1, 3, 3)
# plt.title("Final Residual")
# # plt.title("Final Fractional Residual")
# plt.imshow((final_psfs[0] - data[0])*1e-3)
# # plt.imshow(data[0]/final_psfs[0])
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# divider = make_axes_locatable(ax3)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = plt.colorbar(cax=cax)
# cbar.set_label("Counts $x10^3$")

# # inset axes....
# axins = ax3.inset_axes([0.5, 0.5, 0.45, 0.45])
# axins.imshow((final_psfs[0] - data[0])*1e-3)
# # axins.imshow(data[0]/final_psfs[0])
# # sub region of the original image
# axins.set_xlim(400, 430)
# axins.set_ylim(400-30, 400)
# axins.set_xticklabels([])
# axins.set_yticklabels([])
# ax3.indicate_inset_zoom(axins, edgecolor="black")




plt.tight_layout()
# plt.savefig(paths.figures / "data_resid.pdf", dpi=300)
plt.savefig(paths.figures / "data.pdf", dpi=300)
