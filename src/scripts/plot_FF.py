import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import dill as p
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

pr_true_sort = np.load(paths.data / 'optimise/true_prf_sorted.npy')
pr_found_sort = np.load(paths.data / 'optimise/found_prf_sorted.npy')
colours = np.load(paths.data / 'optimise/colours.npy')

plt.figure(figsize=(10, 4))
plt.suptitle("Pixel Response Function Recovery", size=15)

# print("Starting Scatter Plot")
ax = plt.subplot(1, 2, 1)
plt.scatter(pr_true_sort, pr_found_sort, c=colours*1e-3, alpha=0.5, rasterized=True)
cbar = plt.colorbar()
cbar.set_label("Counts (Photons $x10^3$)")
plt.title("PRF Correlation")
plt.ylabel("Recovered")
plt.xlabel("True")
# print("Finished Scatter Plot")

xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.set_xlim(xlims)
ax.set_ylim(ylims)
plt.plot(np.linspace(0.7, 1.3), np.linspace(0.7, 1.3), c='k', alpha=0.5)

# thresh_indx = np.where(fmask)
# res = (pix_response - flatfields_found[-1])[thresh_indx].flatten()

counts = np.load(paths.data / "optimise/pixel_response_resid_counts.npy")
bins = np.load(paths.data / "optimise/pixel_response_resid_bins.npy")
ax2 = plt.subplot(1, 2, 2)
# plt.hist(res, bins=51)
plt.hist(bins[:-1], bins, weights=counts)
plt.title("PRF Residual Histogram")
plt.ylabel("Counts")
plt.xlabel("Residual")

xlim = np.abs(np.array(ax2.get_xlim())).max()
ax2.set_xlim(-xlim, xlim)

plt.tight_layout()
plt.savefig(paths.figures / "ff.pdf", dpi=300)