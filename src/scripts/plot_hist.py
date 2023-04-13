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
tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
# models_out = p.load(open(paths.data / 'models_out.p', 'rb'))
# losses = np.load(paths.data / 'losses.npy')
data = np.load(paths.data / "make_model_and_data/data.npy")
# psfs_out = np.load(paths.data / "final_psfs.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

# positions_found  = np.array([model.get(positions) for model in models_out])
# fluxes_found     = np.array([model.get(fluxes)    for model in models_out])
# zernikes_found   = np.array([model.get(zernikes)  for model in models_out])
# flatfields_found = np.array([model.get(flatfield) for model in models_out])
positions_found = np.load(paths.data / "optimise/positions_found.npy")
fluxes_found = np.load(paths.data / "optimise/fluxes_found.npy")
zernikes_found = np.load(paths.data / "optimise/zernikes_found.npy")
flatfields_found = np.load(paths.data / "optimise/flatfields_found.npy")

# colours = np.load(paths.data / 'optimise/colours.npy')
# print(colours)

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
thresh = 1000
data_sort = np.sort(data.flatten())
nhists = 4
size = len(data_sort)//nhists

threshes = [0]
for i in range(nhists):
    threshes.append(data_sort[size*i])
threshes.append(data_sort[-1] + 1)

threshes = np.linspace(0, data.max(), nhists+1).astype(int)


data_flat = data.flatten()
indexes = []
for i in range(len(threshes)-1):
    low = np.where(data_flat >= threshes[i])[0]
    high = np.where(data_flat < threshes[i+1])[0]
    indexes.append(np.intersect1d(low, high))

bin_size = []
for i in range(len(indexes)):
    # print(len(indexes[i]))
    # print(len(indexes[i])/2/5)
    bin_size.append(int(np.minimum(np.ceil(len(indexes[i])/25), 50)))
print(bin_size)


import matplotlib
cmap = matplotlib.cm.get_cmap('inferno')

vmin, vmax = data.min(), data.max()
prf_flat = (pix_response - flatfields_found[-1]).flatten()

for i in range(len(indexes)):
    lab = "{} - {}".format(int(threshes[i]), int(threshes[i+1]))

    mean = (threshes[i] + threshes[i+1])/2
    # mean = threshes[i+1]
    # mean = threshes[i]
    mean_norm = (mean - vmin)/(vmax - vmin)
    cols = cmap(mean_norm)[:3]

    plt.hist(prf_flat[indexes[i]], bins=bin_size[i], alpha=0.5, label=lab, color=cols, density=True)
    # plt.hist(prf_flat[indexes[i]], bins=50, alpha=0.85, label=lab, color=cols, density=True)
    # plt.hist(prf_flat[indexes[i]], bins=50, alpha=0.85, density=True)
# cbar = plt.colorbar()
plt.legend()
plt.xlim(-0.2, 0.2)
plt.savefig(paths.figures / "hists.pdf", dpi=300)