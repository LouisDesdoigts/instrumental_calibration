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
models_out = p.load(open(paths.data / 'optimise/models_out.p', 'rb'))
losses = np.load(paths.data / 'optimise/losses.npy')
data = np.load(paths.data / "make_model_and_data/data.npy")
# psfs_out = np.load(paths.data / "optimise/final_psfs.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

# Get parameters
positions_found  = np.array([model.get(positions) for model in models_out])
fluxes_found     = np.array([model.get(fluxes)    for model in models_out])
zernikes_found   = np.array([model.get(zernikes)  for model in models_out])
flatfields_found = np.array([model.get(flatfield) for model in models_out])

# Get the residuals
coeff_residuals     = tel.get(zernikes)  - zernikes_found
flux_residuals      = tel.get(fluxes)    - fluxes_found
flatfield_residuals = tel.get(flatfield) - flatfields_found

scaler = 1e3
positions_residuals = tel.get(positions) - positions_found
r_residuals_rads = np.hypot(positions_residuals[:, :, 0], positions_residuals[:, :, 1])
r_residuals = r2a(r_residuals_rads)

# Positions
arcsec = r2a(1)
true_positions = tel.get(positions).flatten() * arcsec
initial_positions = positions_found[0].flatten() * arcsec
final_positions = positions_found[-1].flatten() * arcsec

plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.title("Log10 Loss")
plt.xlabel("Epochs")
plt.ylabel("Log10 ADU")
plt.plot(np.log10(np.array(losses)))

plt.subplot(3, 2, 2)
plt.title("Stellar Positions")
plt.xlabel("Epochs")
plt.ylabel("Positional Error (arcseconds)")
plt.plot(r_residuals)
plt.axhline(0, c='k', alpha=0.5)

plt.subplot(3, 2, 3)
plt.title("Stellar Fluxes")
plt.xlabel("Epochs")
plt.ylabel("Flux Error (Photons)")
plt.plot(flux_residuals)
plt.axhline(0, c='k', alpha=0.5)

plt.subplot(3, 2, 4)
plt.title("Zernike Coeff Residuals")
plt.xlabel("Epochs")
plt.ylabel("Residual Amplitude")
plt.plot(coeff_residuals)
plt.axhline(0, c='k', alpha=0.5)

plt.subplot(3, 2, 5)
plt.title("PRF")
plt.xlabel("Epochs")
plt.ylabel("SSE")
plt.plot(np.square(flatfield_residuals).sum((-1, -2)))
plt.axhline(0, c='k', alpha=0.5)

# FF Scatter Plot
colours = data.flatten()
ind = np.argsort(colours)
colours = colours[ind]

pr_true_flat = tel.get(flatfield).flatten()[ind]
pr_found_flat = flatfields_found[-1].flatten()[ind]

plt.subplot(3, 2, 6)
plt.title("PRF")
plt.xlabel("True")
plt.ylabel("Recovered")
plt.plot([pr_true_flat.min(), pr_true_flat.max()], [pr_true_flat.min(), pr_true_flat.max()], c='k')
plt.scatter(pr_true_flat, pr_found_flat, c=colours, rasterized=True)
plt.colorbar()

plt.tight_layout()
# plt.savefig('test')

plt.savefig(paths.figures / "progress.pdf", dpi=300)
