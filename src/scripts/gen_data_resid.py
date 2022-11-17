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

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Load model
tel = p.load(open(paths.data / 'instrument.p', 'rb'))
source = p.load(open(paths.data / 'source.p', 'rb'))

# Observe!
psfs = tel.observe()

# Define Params
positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

det_pixsize = tel.get("AngularMFT.pixel_scale_out")
det_npix = tel.get("AngularMFT.npixels_out")
nzern = len(tel.get(zernikes))
Nstars = len(source.get('flux'))

# Add small random values to the positions
model = tel.add(positions, 2.5*det_pixsize*jr.normal(jr.PRNGKey(0),  (Nstars, 2)))

# Multiply the fluxes by small random values
model = model.multiply(fluxes, 1 + 0.1*jr.normal(jr.PRNGKey(0), (Nstars,)))

# Set the zernike coefficients to zero
model = model.set(zernikes, np.zeros(nzern))

# Set the flat fiel to uniform
model = model.set(flatfield, np.ones((det_npix, det_npix)))


# Save optimisation model
p.dump(model, open(paths.data / 'model.p', 'wb'))


# Apply some noise to the PSF Background noise
BG_noise = np.abs(5*jr.normal(jr.PRNGKey(0), psfs.shape))
data = jr.poisson(jr.PRNGKey(0), psfs) + BG_noise
np.save(paths.data / "data", data)

plt.figure(figsize=(15, 4.5))
plt.suptitle("Data and Residuals", size=15)

plt.subplot(1, 3, 1)
plt.title("Sample Data")
plt.imshow(data[0]*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Counts $x10^3$")


initital_psfs = model.observe()
plt.subplot(1, 3, 2)
plt.title("Initial Residual")
plt.imshow((initital_psfs[0] - data[0])*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Counts $x10^3$")


final_model = p.load(open(paths.data / 'models_out.p', 'rb'))[-1]
final_psfs = final_model.observe()

ax = plt.subplot(1, 3, 3)
plt.title("Final Residual")
plt.imshow((final_psfs[0] - data[0])*1e-3)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Counts $x10^3$")

# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
axins.imshow((final_psfs[0] - data[0])*1e-3)
# sub region of the original image
axins.set_xlim(400, 430)
axins.set_ylim(400-30, 400)
axins.set_xticklabels([])
axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="black")


plt.tight_layout()
plt.savefig(paths.figures / "data_resid.pdf", dpi=300)
