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




'''Create Optics, Detector, Source & Instrument'''
# Define wavelengths
wavels = 1e-9 * np.linspace(545, 645, 3)
np.save(paths.data / 'wavelengths.npy', wavels)

# Basic Optical Parameters
aperture = 0.5
wf_npix = 512

# Detector Parameters
det_npix = 1024
sampling_rate = 20
det_pixsize = dl.utils.get_pixel_scale(sampling_rate, wavels.mean(), aperture)

# Load mask
raw_mask = dl.utils.phase_to_opd(np.load("mask.npy"), wavels.mean())
mask = dl.utils.scale_array(raw_mask, wf_npix, 0)

# Zernike Basis
zern_basis = dl.utils.zernike_basis(10, wf_npix, outside=0.)[3:]
coeffs = 2e-8 * jr.normal(jr.PRNGKey(0), [len(zern_basis)])

# Define Optical Configuration
optical_layers = [
    dl.CreateWavefront    (wf_npix, aperture),
    dl.CompoundAperture   ([aperture/2], occulter_radii=[aperture/10]),
    dl.ApplyBasisOPD      (zern_basis, coeffs),
    dl.AddOPD             (mask),
    dl.NormaliseWavefront (),
    dl.AngularMFT         (det_npix, det_pixsize)]

# Create Optics object
optics = dl.Optics(optical_layers)

# Pixel response
pix_response = 1 + 0.05*jr.normal(jr.PRNGKey(0), [det_npix, det_npix])

# Create Detector object
detector = dl.Detector([dl.ApplyPixelResponse(pix_response)])

# Observation stratergy, define dithers
dithers = 2**-.5 * det_pixsize * np.array([[+1, +1],
                                           [+1, -1],
                                           [-1, +1],
                                           [-1, -1]])

def observe_fn(model, dithers):
    return model.dither_and_model(dithers)

# Observation dictionary
observation = {'fn': observe_fn, 'args': dithers}

# Multiple sources to observe
Nstars = 20
true_positions = a2r(jr.uniform(jr.PRNGKey(0), (Nstars, 2), minval=-5, maxval=5))
true_fluxes = 1e8 + 1e7*jr.normal(jr.PRNGKey(0), (Nstars,))

# Create Source object
source = dl.MultiPointSource(true_positions, true_fluxes, wavelengths=wavels)

# Combine into instrument
tel = dl.Instrument(optics=optics, sources=[source], detector=detector,
                    observation=observation)


# Save model
p.dump(tel, open(paths.data / "instrument.p", 'wb'))
p.dump(source, open(paths.data / "source.p", 'wb'))