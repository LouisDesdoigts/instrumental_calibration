import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import pickle as p
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zodiax.experimental.serialisation import serialise, deserialise
from observation import MultiImage

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

flux = 1e8
PRFdev = 1e-1
sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

# Load model
tel = deserialise(paths.data / f'{sub_dir}/instrument.zdx')
model = deserialise(paths.data / f'{sub_dir}/final_model.zdx')

# OPDs
true_opd = tel.Aberrations.get_opd()
found_opd = model.Aberrations.get_opd()
opd_residuls = true_opd - found_opd
opd_rmse_nm = 1e9*np.mean(opd_residuls**2, axis=(-1,-2))**0.5

vmin = np.min(np.array([true_opd, found_opd]))
vmax = np.max(np.array([true_opd, found_opd]))

# Coefficients
zernikes   = 'Aberrations.coefficients'
true_coeff = tel.get(zernikes)
found_coeff = model.get(zernikes)
index = np.arange(len(true_coeff))+4

# Errors
Nzern = len(true_coeff)
errs = np.abs(np.diag(np.load(paths.data / f'{sub_dir}/cov_mat.npy')))**0.5
zerr = errs[-Nzern:]

plt.figure(figsize=(15, 4))
plt.suptitle("Optical Aberrations", size=15)

ax = plt.subplot(1, 3, 1)
divider = make_axes_locatable(ax)

ax2 = divider.new_vertical(size="150%", pad=0.00)
fig1 = ax.get_figure()
fig1.add_axes(ax2)

plt.title("Zernike Coefficients (nm)")
ax2.errorbar(true_coeff*1e9, found_coeff*1e9, yerr=zerr*1e9, fmt='o', capsize=5)
ax2.set_xticks([])
ax2.set_ylabel("Recovered")
xlims = ax2.get_xlim()
ylims = ax2.get_ylim()

vals = 1e9*np.linspace(1.2*true_coeff.min(), 1.2*true_coeff.max(), 2)
ax2.plot(vals, vals, c='k', alpha=0.5)
ax2.set_xlim(xlims)
ax2.set_ylim(ylims)
ax.errorbar(true_coeff*1e9, (true_coeff - found_coeff)*1e9, yerr=zerr*1e9, fmt='o', capsize=5)
ax.axhline(0, c='k')
ax.set_ylabel("Residuals")
ax.set_xlabel("True")
ax.set_ylim(1.1 * np.array(ax.get_ylim()))

throughput = tel.CircularAperture.aperture
true_opd = true_opd.at[np.where(throughput==0.)].set(np.nan)
found_opd = found_opd.at[np.where(throughput==0.)].set(np.nan)

true_vales = true_opd[np.where(throughput==1.)]
found_vales = found_opd[np.where(throughput==1.)]
residual_vales = (true_opd-found_opd)[np.where(throughput==1.)]

true_rms = (true_vales**2).mean()**0.5
found_rms = (found_vales**2).mean()**0.5
residual_rms = (residual_vales**2).mean()**0.5

# from matplotlib.cm import get_cmap
from matplotlib import colormaps as cm
cmap = cm["inferno"]
cmap.set_bad('k',1.)

plt.subplot(1, 3, 2)
plt.title("Recovered OPD: {:.3} nm RMS".format(found_rms*1e9))
plt.imshow(found_opd*1e9, cmap=cmap)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.subplot(1, 3, 3)
plt.title("OPD Residual: {:.3} nm RMS".format(residual_rms*1e9))
plt.imshow((true_opd - found_opd)*1e9, cmap=cmap)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.tight_layout()
plt.savefig(paths.figures / "aberrations.pdf", dpi=300)