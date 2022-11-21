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
from tqdm import tqdm

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Load model
tel = p.load(open(paths.data / 'instrument.p', 'rb'))
models_out = p.load(open(paths.data / 'models_out.p', 'rb'))
losses = np.load(paths.data / 'losses.npy')
data = np.load(paths.data / "data.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]



# Get parameters
nepochs = len(models_out)
psfs_out = models_out[-1].observe()

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
# thresh = 1000
thresh = 500
fmask = data.mean(0) >= thresh

out_mask = np.where(data.mean(0) < thresh)
in_mask = np.where(data.mean(0) >= thresh)

data_tile = np.tile(data.mean(0), [len(models_out), 1, 1])
in_mask_tiled = np.where(data_tile >= thresh)

# calculate residuals
pr_residuals = pix_response[in_mask] - flatfields_found[-1][in_mask]

# for correlation plot
true_pr_masked = pix_response.at[out_mask].set(1)
found_pr_masked = flatfields_found[-1].at[out_mask].set(1)

# FF Scatter Plot
data_sum = data.sum(0) # [flux_mask]
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

# plt.subplot(2, 3, (1,2))
# plt.title("Pixel Response")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Sensitivity Error")
# plt.plot(masked_error)
# plt.axhline(0, c='k', alpha=0.5)

# FF Scatter Plot
data_sum = data.sum(0)
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

thresh_indx = np.where(fmask)
res = (pix_response - flatfields_found[-1])[thresh_indx].flatten()

ax2 = plt.subplot(1, 2, 2)
# plt.plot(np.linspace(0.8, 1.2), np.linspace(0.8, 1.2), c='k', alpha=0.75)
# plt.hist((pr_true_sort - pr_found_sort).flatten(), bins=25)
# plt.hist((pr_true_sort - pr_found_sort).flatten(), bins=51)
# plt.hist((pr_true_sort - pr_found_sort).flatten(), bins=101)
# plt.hist((pr_true_sort - pr_found_sort).flatten(), bins=201)
plt.hist(res, bins=51)
plt.title("PRF Residual Histogram")
plt.ylabel("Counts")
plt.xlabel("Residual")

xlim = np.abs(np.array(ax2.get_xlim())).max()
ax2.set_xlim(-xlim, xlim)

plt.xticks(np.linspace(-0.1, 0.1, 5))

# for index, label in enumerate(ax2.xaxis.get_ticklabels()):
#     if (index+1) % 2 != 0:
#         label.set_visible(False)
#     print(index, label, label.get_visible())

# for index, label in enumerate(ax2.xaxis.get_ticklabels()):
#     print(index, label, label.get_visible())

# for index, label in enumerate(ax2.xaxis.get_ticklabels()):
#     if (index+1) % 2 != 0:
#         label.set_visible(False)



# plt.subplot(1, 3, 3)
# plt.hist(np.abs(res).flatten(), bins=100)
# plt.semilogx()
# plt.title("PRF Residual Histogram")
# plt.ylabel("Counts")
# plt.xlabel("Residual")

# plt.subplot(2, 3, 4)
# plt.title("True Pixel Response")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.imshow(true_pr_masked)
# plt.colorbar()

# vmin = np.min(pix_response)
# vmax = np.max(pix_response)

# plt.subplot(2, 3, 5)
# plt.title("Found Pixel Response")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.imshow(found_pr_masked, vmin=vmin, vmax=vmax)
# plt.colorbar()

# plt.subplot(2, 3, 6)
# plt.title("Pixel Response Residual")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.imshow(true_pr_masked - found_pr_masked, vmin=-0.2, vmax=0.2)
# plt.colorbar()

plt.tight_layout()
plt.savefig(paths.figures / "ff.pdf", dpi=300)