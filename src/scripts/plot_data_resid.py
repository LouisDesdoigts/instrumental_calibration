import jax.numpy as np
import paths
import pickle as p
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zodiax.experimental.serialisation import serialise, deserialise
from observation import MultiImage

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

flux = 1e8
PRFdev = 1e-1
sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

# Load model
data = np.load(paths.data / f"{sub_dir}/data.npy")
tel = deserialise(paths.data / f"{sub_dir}/instrument.zdx")
model = deserialise(paths.data / f"{sub_dir}/initial_model.zdx")
final_model = deserialise(paths.data / f"{sub_dir}/final_model.zdx")

### Full PSFs ###
initial_psfs = model.observe()
final_psfs = final_model.observe()

plt.figure(figsize=(15, 4.5))
plt.suptitle("Data and Residuals", size=15)

ax1 = plt.subplot(1, 3, 1)
rms_data = np.sqrt((data**2).mean())
plt.title(f"Data, Mean counts: {data.mean():.0f}")
plt.imshow(data.sum(0)*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")

ax2 = plt.subplot(1, 3, 2)
initial_rms = np.sqrt(((initial_psfs - data)**2).mean())
plt.title(f"Initial Residual: {initial_rms:.0f} counts RMS")
plt.imshow((initial_psfs.sum(0) - data.sum(0))*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts $x10^3$")

ax3 = plt.subplot(1, 3, 3)
final_rms = np.sqrt(((final_psfs - data)**2).mean())
plt.title(f"Final Residual: {final_rms:.0f} counts RMS")
plt.imshow((final_psfs.sum(0) - data.sum(0)))
plt.xlabel("Pixels")
plt.ylabel("Pixels")
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(cax=cax)
cbar.set_label("Counts")

# inset axes....
axins = ax3.inset_axes([0.5, 0.5, 0.45, 0.45])
print("TODO: Check that this should be *1e-3")
axins.imshow((final_psfs.sum(0) - data.sum(0))*1e-3)
# sub region of the original image
axins.set_xlim(400, 430)
axins.set_ylim(400-30, 400)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax3.indicate_inset_zoom(axins, edgecolor="black")

plt.tight_layout()
plt.savefig(paths.figures / "data_resid.pdf", dpi=300)
