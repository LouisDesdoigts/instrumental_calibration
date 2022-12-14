import jax.numpy as np
import paths
import pickle as p
import dLux as dl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

# Load model
tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
wavels = np.load(paths.data / 'make_model_and_data/wavelengths.npy')
psf = np.load(paths.data / "make_model_and_data/plain_psf.npy")
aberrated_psf = np.load(paths.data / "make_model_and_data/aberrated_psf.npy")

# Pupil
throughput = tel.CompoundAperture.get_aperture(npixels=tel.get('CreateWavefront.npixels'))
mask = tel.AddOPD.opd
pupil = mask.at[np.where(throughput==0.)].set(np.nan)

# Aberrations
opd = tel.ApplyBasisOPD.get_total_opd()
aberrated_pupil = pupil + opd

# FF
FF = tel.ApplyPixelResponse.pixel_response

cmap = get_cmap("inferno").copy()
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
plt.imshow(psf)
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

counts = np.load(paths.data / "make_model_and_data/pixel_response_counts.npy")
bins = np.load(paths.data / "make_model_and_data/pixel_response_bins.npy")
plt.subplot(2, 3, 3)
plt.title("Pixel Response Distribution")
plt.hist(bins[:-1], bins, weights=counts)
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