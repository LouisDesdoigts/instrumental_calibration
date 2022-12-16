import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import pickle as p
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

ax = plt.subplot(1, 2, 1)
plt.scatter(pr_true_sort, pr_found_sort, c=colours*1e-3, alpha=0.5, rasterized=True)
cbar = plt.colorbar()
cbar.set_label("Counts (Photons $x10^3$)")
plt.title("PRF Correlation")
plt.ylabel("Recovered")
plt.xlabel("True")

xlims = ax.get_xlim()
ylims = ax.get_ylim()
ax.set_xlim(xlims)
ax.set_ylim(ylims)
plt.plot(np.linspace(0.7, 1.3), np.linspace(0.7, 1.3), c='k', alpha=0.5)

counts = np.load(paths.data / "optimise/pixel_response_resid_counts.npy")
bins = np.load(paths.data / "optimise/pixel_response_resid_bins.npy")
ax2 = plt.subplot(1, 2, 2)
plt.hist(bins[:-1], bins, weights=counts)
plt.title("PRF Residual Histogram")
plt.ylabel("Counts")
plt.xlabel("Residual")

xlim = np.abs(np.array(ax2.get_xlim())).max()
ax2.set_xlim(-xlim, xlim)


# Plot
# calculate the mask where there was enough flux to infer the flat field

# data = np.load(paths.data / "make_model_and_data/data.npy")
# flatfields_found = np.load(paths.data / "optimise/flatfields_found.npy")
# tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
# pix_response = tel.get('ApplyPixelResponse.pixel_response')

# thresh = 1000
# data_sort = np.sort(data.flatten())
# nhists = 4
# size = len(data_sort)//nhists

# threshes = [0]
# for i in range(nhists):
#     threshes.append(data_sort[size*i])
# threshes.append(data_sort[-1] + 1)

# data_flat = data.flatten()
# indexes = []
# for i in range(len(threshes)-1):
#     low = np.where(data_flat >= threshes[i])[0]
#     high = np.where(data_flat < threshes[i+1])[0]
#     indexes.append(np.intersect1d(low, high))

# import matplotlib
# cmap = matplotlib.cm.get_cmap('inferno')

# im = ax2.imshow([[0., 1.]])#, vmin=data/1e3, vmax=data/1e3, cmap='inferno')
# im.set_visible(False)
# # ax2.gca.set_visible(False)


# vmin, vmax = data.min(), data.max()
# prf_flat = (pix_response - flatfields_found[-1]).flatten()

# for i in range(len(indexes)):
#     lab = "{} - {}".format(int(threshes[i]), int(threshes[i+1]))

#     # mean = (threshes[i] + threshes[i+1])/2
#     mean = threshes[i+1]
#     mean_norm = (mean - vmin)/(vmax - vmin)
#     cols = cmap(mean_norm)[:3]

#     ax2.hist(prf_flat[indexes[i]], bins=50, alpha=0.85, label=lab, color=cols)
# cbar = plt.colorbar()
# ax2.legend()
# ax2.set_xlim(-0.2, 0.2)

plt.tight_layout()
plt.savefig(paths.figures / "ff.pdf", dpi=300)