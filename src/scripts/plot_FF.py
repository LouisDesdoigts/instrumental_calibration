import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import matplotlib.pyplot as plt
from zodiax.experimental.serialisation import serialise, deserialise
from observation import MultiImage
import jax.scipy as jsp
from lib import weighted_pearson_corr_coeff

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

flux = 1e8
PRFdev = 1e-1
sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

# Load data
tel = deserialise(paths.data / f'{sub_dir}/instrument.zdx')
model = deserialise(paths.data / f'{sub_dir}/final_model.zdx')
data = np.load(paths.data / f'{sub_dir}/data.npy')
flatfield = 'PRF.pixel_response'

PRF_true = tel.get(flatfield)
PRF_found = model.get(flatfield)
PRF_residual = PRF_true - PRF_found

colours = data.sum(0).flatten()
ind = np.argsort(colours)
colours = colours[ind]
PRF_true_sort = PRF_true.flatten()[ind]
PRF_found_sort = PRF_found.flatten()[ind]

corr = weighted_pearson_corr_coeff(PRF_true, PRF_found, data.sum(0))

# Plot
plt.figure(figsize=(10, 8))
plt.suptitle("Pixel Response Function Recovery", size=15)

ax = plt.subplot(2, 2, 1)
plt.title(f"PRF Correlation: {corr:.3f}")
plt.ylabel("Recovered")
plt.xlabel("True")
plt.scatter(PRF_true_sort, PRF_found_sort, c=colours*1e-3, alpha=0.5, rasterized=True)
cbar = plt.colorbar()
cbar.set_label("Counts (Photons $x10^3$)")
xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.set_xlim(xlims)
ax.set_ylim(ylims)
plt.plot(np.linspace(0.7, 1.3), np.linspace(0.7, 1.3), c='k', alpha=0.5)

plt.subplot(2, 2, 2)
plt.imshow(PRF_found)
plt.colorbar()
plt.title("Recovered PRF")
plt.xlabel("Pixles")
plt.ylabel("Pixles")

ax2 = plt.subplot(2, 2, 3)
plt.hist(PRF_residual.flatten(), bins=51)
plt.title("PRF Residual Histogram")
plt.ylabel("Occurrences")
plt.xlabel("Residual")

xlim = np.abs(np.array(ax2.get_xlim())).max()
ax2.set_xlim(-xlim, xlim)

ax3 = plt.subplot(2, 2, 4)
ax3.hist(PRF_found.flatten(), bins=51)
plt.title("Recovered PRF Histogram")
plt.ylabel("Occurrences")
plt.xlabel("Value")

xs = 1 + np.linspace(-4*PRFdev, 4*PRFdev)
prob = jsp.stats.norm.pdf(xs, loc=1, scale=PRFdev)
ax4 = ax3.twinx()
ax4.plot(xs, prob, c='r', label='True Distribution')
ax4.set_ylim(0)
ax4.set_yticks([])
plt.legend()

plt.tight_layout()
plt.savefig(paths.figures / f"ff.pdf", dpi=300)