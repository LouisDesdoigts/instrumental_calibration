import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import pickle as p
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

# Load model
tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
positions_found = np.load(paths.data / "optimise/positions_found.npy")
fluxes_found = np.load(paths.data / "optimise/fluxes_found.npy")
zernikes_found = np.load(paths.data / "optimise/zernikes_found.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

# Get the residuals
coeff_residuals = tel.get(zernikes) - zernikes_found
flux_residuals = tel.get(fluxes) - fluxes_found

scaler = 1e3
positions_residuals = tel.get(positions) - positions_found
r_residuals_rads = np.hypot(positions_residuals[:, :, 0], positions_residuals[:, :, 1])
r_residuals = r2a(r_residuals_rads)

# Errors
Nstars = len(tel.get(positions))
errs = np.diag(np.load(paths.data / 'calc_errors/cov_mat.npy'))**0.5

# Positions
arcsec = r2a(1)
true_positions = tel.get(positions).flatten() * arcsec
initial_positions = positions_found[0].flatten() * arcsec
final_positions = positions_found[-1].flatten() * arcsec
pos_err = errs[:2*Nstars] * arcsec

from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(figsize=(10, 4))

plt.suptitle("Astrophysical Parameters", size=15)

ax1 = plt.subplot(1, 2, 1)
divider = make_axes_locatable(ax1)
ax2 = divider.new_vertical(size="150%", pad=0.00)
fig.add_axes(ax2)
ax2.set_title("Positions (arcseconds)")
ax2.errorbar(true_positions, final_positions, yerr=pos_err, fmt='o', capsize=5)

xlims = ax2.get_xlim()
ylims = ax2.get_ylim()

vals = np.linspace(1.2*true_positions.min(), 1.2*true_positions.max(), 2)
ax2.plot(vals, vals, c='k', alpha=0.5)
ax2.set_ylabel('Recovered')
ax2.set_xticks([])
ax2.set_xlim(xlims)
ax2.set_ylim(ylims)

ax1.errorbar(true_positions, true_positions-final_positions, yerr=pos_err, fmt='o', capsize=5)
ax1.axhline(0, c='k', alpha=0.5)
ax1.set_xlabel('True')
ax1.set_ylabel('Residual')
ax1.set_ylim(1.1 * np.array(ax1.get_ylim()))

# Fluxes
true_fluxes = tel.get(fluxes) * 1e-6
final_fluxes = fluxes_found[-1] * 1e-6
initial_fluxes = fluxes_found[0] * 1e-6
flux_err = errs[2*Nstars:3*Nstars] * 1e-6
flux_err_phot = fluxes_found[-1]**0.5 * 1e-6

ax3 = plt.subplot(1, 2, 2)
divider = make_axes_locatable(ax3)
ax4 = divider.new_vertical(size="150%", pad=0.00)
fig.add_axes(ax4)
ax4.set_title("Fluxes (photons) $x10^6$")
ax4.errorbar(true_fluxes, final_fluxes, yerr=flux_err, fmt='o', capsize=5)

xlims = ax4.get_xlim()
ylims = ax4.get_ylim()

vals = np.linspace(0.8*true_fluxes.min(), 1.2*true_fluxes.max(), 2)
ax4.plot(vals, vals, c='k', alpha=0.5)
ax4.set_xticks([])
ax4.set_xlim(xlims)
ax4.set_ylim(ylims)
ax4.set_ylabel('Recovered')
ax4.set_xticks([])

ax3.errorbar(true_fluxes, true_fluxes-final_fluxes, yerr=flux_err, fmt='o', capsize=5)
ax3.axhline(0, c='k', alpha=0.5)
ax3.set_xlabel('True')
ax3.set_ylabel('Residual')
ax3.set_ylim(1.1 * np.array(ax3.get_ylim()))

plt.tight_layout()
plt.savefig(paths.figures / "astro_params.pdf", dpi=300)