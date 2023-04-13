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
tel = deserialise(paths.data / f'make_model_and_data/{sub_dir}/instrument.zdx')
model = deserialise(paths.data / f'optimise/{sub_dir}/final_model.zdx')
data = np.load(paths.data / f'make_model_and_data/{sub_dir}/data.npy')
params = deserialise(paths.data / f"make_model_and_data/{sub_dir}/params.zdx")

# PRFs
pr_true_sort = np.load(paths.data / f'process/{sub_dir}/true_prf_sorted.npy')
pr_found_sort = np.load(paths.data / f'process/{sub_dir}/found_prf_sorted.npy')
colours = np.load(paths.data / f'process/{sub_dir}/colours.npy')

# Histograms
counts = np.load(paths.data / f"process/{sub_dir}/PRF_found_counts.npy")
bins = np.load(paths.data / f"process/{sub_dir}/PRF_found_bins.npy")
resid_counts = np.load(paths.data / f"process/{sub_dir}/PRF_resid_counts.npy")
resid_bins = np.load(paths.data / f"process/{sub_dir}/PRF_resid_bins.npy")

# Correlation coefficient
true_prf = model.PRF.pixel_response
found_prf = tel.PRF.pixel_response
corr = weighted_pearson_corr_coeff(found_prf, true_prf, data.sum(0))


# Plot
plt.figure(figsize=(10, 8))
plt.suptitle("Pixel Response Function Recovery", size=15)

ax = plt.subplot(2, 2, 1)
plt.scatter(pr_true_sort, pr_found_sort, c=colours*1e-3, alpha=0.5, rasterized=True)
cbar = plt.colorbar()
cbar.set_label("Counts (Photons $x10^3$)")
plt.title(f"PRF Correlation: {corr:.3f}")
plt.ylabel("Recovered")
plt.xlabel("True")

xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.set_xlim(xlims)
ax.set_ylim(ylims)
plt.plot(np.linspace(0.7, 1.3), np.linspace(0.7, 1.3), c='k', alpha=0.5)


plt.subplot(2, 2, 2)
plt.imshow(found_prf)
plt.colorbar()
plt.title("Recovered PRF")
plt.xlabel("Pixles")
plt.ylabel("Pixles")

ax2 = plt.subplot(2, 2, 3)
plt.hist(resid_bins[:-1], resid_bins, weights=resid_counts)
plt.title("PRF Residual Histogram")
plt.ylabel("Counts")
plt.xlabel("Residual")

xlim = np.abs(np.array(ax2.get_xlim())).max()
ax2.set_xlim(-xlim, xlim)

ax3 = plt.subplot(2, 2, 4)
try:
    ax3.hist(bins[:-1], bins, weights=counts)
except ValueError:
    ax3.hist(pr_found_sort, bins=51)
plt.title("Recovered PRF Histogram")
plt.ylabel("Counts")
plt.xlabel("Value")

PRdev = params['PRFdev']
xs = 1 + np.linspace(-4*PRdev, 4*PRdev)
prob = jsp.stats.norm.pdf(xs, loc=1, scale=PRdev)
ax4 = ax3.twinx()
ax4.plot(xs, prob, c='r', label='True Distribution')
ax4.set_ylim(0)
ax4.set_yticks([])
plt.legend()

plt.tight_layout()
plt.savefig(paths.figures / f"ff.pdf", dpi=300)