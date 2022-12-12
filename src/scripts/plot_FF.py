import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import dill as p
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Load model
tel = p.load(open(paths.data / 'instrument.p', 'rb'))
models_out = p.load(open(paths.data / 'models_out.p', 'rb'))
losses = np.load(paths.data / 'losses.npy')
data = np.load(paths.data / "data.npy")
psfs_out = np.load(paths.data / "final_psfs.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]



# # Get parameters
# nepochs = len(models_out)
# # psfs_out = models_out[-1].observe()
# psfs_out = models_out[-1].model()

positions_found  = np.array([model.get(positions) for model in models_out])
fluxes_found     = np.array([model.get(fluxes)    for model in models_out])
zernikes_found   = np.array([model.get(zernikes)  for model in models_out])
flatfields_found = np.array([model.get(flatfield) for model in models_out])


# Get the residuals
coeff_residuals = tel.get(zernikes) - zernikes_found
flux_residuals = tel.get(fluxes) - fluxes_found

scaler = 1e3
positions_residuals = tel.get(positions) - positions_found
r_residuals_rads = np.hypot(positions_residuals[:, :, 0], positions_residuals[:, :, 1])
r_residuals = r2a(r_residuals_rads)


pix_response = tel.get(flatfield)

# Plot
# calculate the mask where there was enough flux to infer the flat field
# thresh = 2500
thresh = 1000
# thresh = 500

# data_sort = np.sort(data[0].flatten())
data_sort = np.sort(data.flatten())
nhists = 4
size = len(data_sort)//nhists

threshes = [0]
for i in range(nhists):
    threshes.append(data_sort[size*i])
threshes.append(data_sort[-1] + 1)

# print(threshes)

# threshes = np.array([0, 2500, 5000, 7500, 10000, 1e6])

# data_flat = data[0].flatten()
data_flat = data.flatten()
indexes = []
for i in range(len(threshes)-1):
    low = np.where(data_flat >= threshes[i])[0]
    high = np.where(data_flat < threshes[i+1])[0]
    indexes.append(np.intersect1d(low, high))
    # intersects.append(len(np.intersect1d(low, high)))
    # print(len(np.intersect1d(low, high)))

# intersects = np.array(intersects)
# print()
# print(intersects.sum())
# print(len(data_flat))

    # print(threshes[i])
    # print(np.where(data_flat >= threshes[i]))
    # print(np.where((data_flat >= threshes[i]) and (data_flat < threshes[i+1])))
    # print()

prf_flat = (pix_response - flatfields_found[-1]).flatten()

# prf_flat = pix_response.flatten()
for i in range(len(indexes)):
    # print(len(indexes[i]))
    lab = "{} - {}".format(int(threshes[i]), int(threshes[i+1]))
    plt.hist(prf_flat[indexes[i]], bins=50, alpha=0.85, label=lab)
plt.legend()
plt.xlim(-0.2, 0.2)
plt.savefig(paths.figures / "hists.pdf", dpi=300)
# print("Done hists")


# fmask = data.mean(0) >= thresh
fmask = data >= thresh
# fmask = data >= thresh

# out_mask = np.where(data.mean(0) < thresh)
out_mask = np.where(data < thresh)
# out_mask = np.where(data < thresh)
# in_mask = np.where(data.mean(0) >= thresh)
in_mask = np.where(data >= thresh)
# in_mask = np.where(data >= thresh)

# data_tile = np.tile(data.mean(0), [len(models_out), 1, 1])
data_tile = np.tile(data, [len(models_out), 1, 1])
# data_tile = np.tile(data, [len(models_out), 1, 1])
in_mask_tiled = np.where(data_tile >= thresh)

# calculate residuals
pr_residuals = pix_response[in_mask] - flatfields_found[-1][in_mask]

# for correlation plot
true_pr_masked = pix_response.at[out_mask].set(1)
found_pr_masked = flatfields_found[-1].at[out_mask].set(1)

# FF Scatter Plot
# data_sum = data.sum(0) # [flux_mask]
data_sum = data # [flux_mask]
# data_sum = data
colours = data_sum.flatten()
ind = np.argsort(colours)
colours = colours[ind]

pr_true_flat = true_pr_masked.flatten()
pr_found_flat = found_pr_masked.flatten()

pr_true_sort = pr_true_flat[ind]
pr_found_sort = pr_found_flat[ind]

# Errors
pfound = flatfields_found[in_mask_tiled].reshape([len(models_out), len(in_mask[0])])
ptrue = pix_response[in_mask]
pr_res = ptrue - pfound
masked_error = np.abs(pr_res).mean(-1)

plt.figure(figsize=(10, 4))
plt.suptitle("Pixel Response Function Recovery", size=15)

# FF Scatter Plot
# data_sum = data.sum(0)
data_sum = data
# data_sum = data
colours = data_sum.flatten()
ind = np.argsort(colours)
colours = colours[ind]

pr_true_flat = true_pr_masked.flatten()
pr_found_flat = found_pr_masked.flatten()

pr_true_sort = pr_true_flat[ind]
pr_found_sort = pr_found_flat[ind]

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

# print(pix_response.shape)
# print(flatfields_found[-1].shape)
# print(thresh_indx.shape)

thresh_indx = np.where(fmask)
res = (pix_response - flatfields_found[-1])[thresh_indx].flatten()

ax2 = plt.subplot(1, 2, 2)
plt.hist(res, bins=51)
plt.title("PRF Residual Histogram")
plt.ylabel("Counts")
plt.xlabel("Residual")

xlim = np.abs(np.array(ax2.get_xlim())).max()
ax2.set_xlim(-xlim, xlim)

# plt.xticks(np.linspace(-0.1, 0.1, 5))


plt.tight_layout()
plt.savefig(paths.figures / "ff.pdf", dpi=300)