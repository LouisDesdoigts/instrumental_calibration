import jax.numpy as np
import paths
import pickle as p
import dLux as dl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
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

### PSFs ###
zernikes = 'Aberrations.coefficients'
wavels = tel.get('Source.spectrum.wavelengths')
aberrated_psf = tel.optics.propagate(wavels)
unaberrated_optics = tel.optics.multiply(zernikes, 0.)
unit_psf = unaberrated_optics.propagate(wavels)

# Diffractive Pupil
support = tel.CircularAperture.aperture
support_mask = np.where(support==0.)
pupil = tel.AddOPD.opd.at[support_mask].set(np.nan)
aberrated_pupil = pupil + tel.Aberrations.get_opd()

# FF
FF = tel.PRF.pixel_response
from matplotlib import colormaps as cm
cmap = cm["inferno"]
cmap.set_bad('k',1.)

plt.figure(figsize=(15, 8))
plt.suptitle("Optical and Instrumental Configuration", size=15)

plt.subplot(2, 3, 1)
plt.title("Pupil OPD")
plt.imshow(pupil * 1e9, cmap=cmap)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.subplot(2, 3, 4)
plt.title("PSF")
plt.imshow(unit_psf)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Probability")

plt.subplot(2, 3, 2)
plt.title("Aberrated Pupil OPD")
plt.imshow(aberrated_pupil * 1e9, cmap=cmap)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.subplot(2, 3, 5)
plt.title("Aberrated PSF")
plt.imshow(aberrated_psf)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Probability")

plt.subplot(2, 3, 3)
plt.title("Pixel Response Distribution")
plt.hist(FF.flatten(), bins=51)
plt.ylabel("Counts")
plt.xlabel("Relative Sensitivity")

ax = plt.subplot(2, 3, 6)
plt.title("Pixel Response")
plt.imshow(FF)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Sensitivity")

# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
axins.imshow(FF)
# sub region of the original image
axins.set_xlim(400, 430)
axins.set_ylim(400-30, 400)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")

plt.tight_layout()
plt.savefig(paths.figures / "optics.pdf", dpi=300)