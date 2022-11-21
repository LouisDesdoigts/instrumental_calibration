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
# models_out = p.load(open(paths.data / 'models_out.p', 'rb'))
# losses = np.load(paths.data / 'losses.npy')
# data = np.load(paths.data / "data.npy")
models_out = p.load(open(paths.data / 'models_out_.p', 'rb'))
losses = np.load(paths.data / 'losses_.npy')
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


# Positions
arcsec = r2a(1)
true_positions = tel.get(positions).flatten() * arcsec
initial_positions = positions_found[0].flatten() * arcsec
final_positions = positions_found[-1].flatten() * arcsec


from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(figsize=(10, 4))

plt.suptitle("Astrophysical Parameters", size=15)

ax1 = plt.subplot(1, 2, 1)
divider = make_axes_locatable(ax1)
ax2 = divider.new_vertical(size="150%", pad=0.00)
fig.add_axes(ax2)
ax2.set_title("Positions (arcseconds)")

ax2.scatter(true_positions, final_positions, label='final')
# ax2.scatter(true_positions, initial_positions, c='tab:orange', marker='x', label="intitial")

xlims = ax2.get_xlim()
ylims = ax2.get_ylim()

vals = np.linspace(1.2*true_positions.min(), 1.2*true_positions.max(), 2)
ax2.plot(vals, vals, c='k', alpha=0.5)
ax2.set_ylabel('Recovered')
ax2.set_xticks([])
ax2.set_xlim(xlims)
ax2.set_ylim(ylims)

ax1.scatter(true_positions, true_positions-final_positions, label='final')
# ax1.scatter(true_positions, true_positions-initial_positions, c='tab:orange', marker='x', label="intitial")
ax1.axhline(0, c='k', alpha=0.5)
ax1.set_xlabel('True')
ax1.set_ylabel('Residual')
ax1.set_ylim(1.1 * np.array(ax1.get_ylim()))


# Fluxes
true_fluxes = tel.get(fluxes) * 1e-6
final_fluxes = fluxes_found[-1] * 1e-6
initial_fluxes = fluxes_found[0] * 1e-6
flux_err = (fluxes_found[-1]**0.5) * 1e-6

# # Fluxes
# true_fluxes = tel.get(fluxes)
# final_fluxes = fluxes_found[-1]
# initial_fluxes = fluxes_found[0]
# flux_err = (fluxes_found[-1]**0.5)


ax3 = plt.subplot(1, 2, 2)
divider = make_axes_locatable(ax3)
ax4 = divider.new_vertical(size="150%", pad=0.00)
fig.add_axes(ax4)
ax4.set_title("Fluxes (photons) $x10^6$")

ax4.scatter(true_fluxes, final_fluxes, label='final')
# ax4.errorbar(true_fluxes, final_fluxes, yerr=flux_err, fmt='o', label='final', capsize=5)
# ax4.scatter(true_fluxes, initial_fluxes, c='tab:orange', marker='x', label="intitial")

xlims = ax4.get_xlim()
ylims = ax4.get_ylim()

vals = np.linspace(0.8*true_fluxes.min(), 1.2*true_fluxes.max(), 2)
ax4.plot(vals, vals, c='k', alpha=0.5)
ax4.set_xticks([])
ax4.set_xlim(xlims)
ax4.set_ylim(ylims)

ax4.set_ylabel('Recovered')
ax4.set_xticks([])
# yticks = ax4.yaxis.get_major_ticks()
# yticks[0].set_visible(False)
# yticks[1].set_visible(False)

ax3.scatter(true_fluxes, true_fluxes-final_fluxes, label='final')
# ax3.errorbar(true_fluxes, true_fluxes-final_fluxes, yerr=flux_err, fmt='o', label='final', capsize=5)
# ax3.scatter(true_fluxes, true_fluxes-initial_fluxes, c='tab:orange', marker='x', label="intitial")

ax3.axhline(0, c='k', alpha=0.5)
ax3.set_xlabel('True')
ax3.set_ylabel('Residual')
ax3.set_ylim(1.1 * np.array(ax3.get_ylim()))


plt.tight_layout()
plt.savefig(paths.figures / "astro_params.pdf", dpi=300)

