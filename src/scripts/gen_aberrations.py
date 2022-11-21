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




# Plot
# OPDs
true_opd = tel.ApplyBasisOPD.get_total_opd()
opds_found = np.array([model.ApplyBasisOPD.get_total_opd() for model in models_out])
found_opd = opds_found[-1]
opd_residuls = true_opd - opds_found
opd_rmse_nm = 1e9*np.mean(opd_residuls**2, axis=(-1,-2))**0.5

vmin = np.min(np.array([true_opd, found_opd]))
vmax = np.max(np.array([true_opd, found_opd]))

# Coefficients
true_coeff = tel.get(zernikes)
found_coeff = models_out[-1].get(zernikes)
index = np.arange(len(true_coeff))+4

plt.figure(figsize=(15, 4))
plt.suptitle("Optical Aberrations", size=15)


from mpl_toolkits.axes_grid1 import make_axes_locatable
ax = plt.subplot(1, 3, 1)
divider = make_axes_locatable(ax)

ax2 = divider.new_vertical(size="150%", pad=0.00)
fig1 = ax.get_figure()
fig1.add_axes(ax2)


plt.title("Zernike Coefficients (nm)")
ax2.scatter(true_coeff*1e9, found_coeff*1e9)

ax2.set_xticks([])
ax2.set_ylabel("Recovered")
# ax2.yaxis.set_tick_params(labelbottom=False)

xlims = ax2.get_xlim()
ylims = ax2.get_ylim()

vals = 1e9*np.linspace(1.2*true_coeff.min(), 1.2*true_coeff.max(), 2)
ax2.plot(vals, vals, c='k', alpha=0.5)

# vals = np.linspace(0.8*true_fluxes.min(), 1.2*true_fluxes.max(), 2)
# ax4.plot(vals, vals, c='k', alpha=0.5)
ax2.set_xlim(xlims)
ax2.set_ylim(ylims)

ax.scatter(true_coeff*1e9, (true_coeff - found_coeff)*1e9)
# ax.scatter(true_coeff, (true_coeff - found_coeff)/true_coeff)
ax.axhline(0, c='k')
ax.set_ylabel("Residuals")
ax.set_xlabel("True")

ax.set_ylim(1.1 * np.array(ax.get_ylim()))



throughput = tel.CompoundAperture.get_aperture(npixels=tel.get('CreateWavefront.npixels'))
mask = tel.AddOPD.opd
true_opd = true_opd.at[np.where(throughput==0.)].set(np.nan)
found_opd = found_opd.at[np.where(throughput==0.)].set(np.nan)

true_vales = true_opd[np.where(throughput==1.)]
found_vales = found_opd[np.where(throughput==1.)]
residual_vales = (true_opd-found_opd)[np.where(throughput==1.)]

true_rms = (true_vales**2).mean()**0.5
found_rms = (found_vales**2).mean()**0.5
residual_rms = (residual_vales**2).mean()**0.5

# Save values
with open(paths.output / "rms_opd_in.txt", 'w') as f:
    f.write("{:.3}".format(true_rms*1e9))

with open(paths.output / "rms_opd_resid.txt", 'w') as f:
    f.write("{:.3}".format(residual_rms*1e9))

from matplotlib.cm import get_cmap
cmap = get_cmap("inferno")
cmap.set_bad('k',1.)

plt.subplot(1, 3, 2)
# plt.title("True OPD: {:.3} nm RMS".format(true_rms*1e9))
# plt.imshow(true_opd*1e9, cmap=cmap)
plt.title("Recovered OPD: {:.3} nm RMS".format(found_rms*1e9))
plt.imshow(found_opd*1e9, cmap=cmap)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.subplot(1, 3, 3)
plt.title("OPD Residual: {:.3} nm RMS".format(residual_rms*1e9))
# plt.imshow((true_opd - found_opd)*1e9, vmin=vmin*1e9, vmax=vmax*1e9, cmap=cmap)
plt.imshow((true_opd - found_opd)*1e9, cmap=cmap)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.tight_layout()
plt.savefig(paths.figures / "aberrations.pdf", dpi=300)
