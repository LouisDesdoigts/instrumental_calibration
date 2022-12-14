# Core jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import dLux as dl
from dLux.utils import arcseconds_to_radians as a2r
import pickle as p
import paths

import os
try:
    os.mkdir(paths.data / "make_model_and_data")
except FileExistsError:
    pass

'''Construct Optics and Detector'''
# Define wavelengths
wavels = 1e-9 * np.linspace(545, 645, 3)
np.save(paths.data / 'make_model_and_data/wavelengths.npy', wavels)

# Basic Optical Parameters
aperture = 0.5
wf_npix = 512

# Detector Parameters
det_npix = 1024
sampling_rate = 20
det_pixsize = dl.utils.get_pixel_scale(sampling_rate, wavels.mean(), aperture)

# Load mask
raw_mask = dl.utils.phase_to_opd(np.load(paths.data / "mask.npy"), wavels.mean())
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
pix_response = 1 + 0.05*jr.normal(jr.PRNGKey(1), [det_npix, det_npix])
counts, bins = np.histogram(pix_response.flatten(), bins=50)
np.save(paths.data / "make_model_and_data/pixel_response_counts.npy", counts)
np.save(paths.data / "make_model_and_data/pixel_response_bins.npy", bins)

# Create detector layers
bg_mean = 10
detector_layers =[dl.ApplyPixelResponse(pix_response),
                  dl.AddConstant(bg_mean)]

# Create Detector object
detector = dl.Detector(detector_layers)


"""Construct sources"""
# Multiple sources to observe
Nstars = 20
flux = 1e8
true_fluxes = flux + (flux/10)*jr.normal(jr.PRNGKey(2), (Nstars,))

# Random
r_max = 4.5
true_positions = a2r(jr.uniform(jr.PRNGKey(4), (Nstars, 2), minval=-r_max, maxval=r_max))

# Create Source object
source = dl.MultiPointSource(true_positions, true_fluxes, wavelengths=wavels)

"""Construct Instrument"""
# Combine into instrument
tel = dl.Instrument(optics=optics, sources=[source], detector=detector)

"""Make some data with photon and detector noise"""
# Observe!
bg_val = tel.AddConstant.value
psfs = tel.set(['detector'], [None]).model()

# Apply some noise to the PSF Background noise
BG_noise = np.maximum(2.5*jr.normal(jr.PRNGKey(3), psfs.shape) + bg_val, 0.)
psf_photon = tel.ApplyPixelResponse.pixel_response * jr.poisson(jr.PRNGKey(5), psfs)
data = psf_photon + BG_noise

# Define Params
positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

"""Perturb model for optimisation"""
# Add small random values to the positions
model = tel.add(positions, 1.*det_pixsize*jr.normal(jr.PRNGKey(6),  (Nstars, 2)))

# Multiply the fluxes by small random values
model = model.multiply(fluxes, 1 + 0.1*jr.normal(jr.PRNGKey(7), (Nstars,)))

# Set the zernike coefficients to zero
model = model.set(zernikes, np.zeros(len(zern_basis)))

# Set the flat fiel to uniform
model = model.set(flatfield, np.ones((det_npix, det_npix)))

"""Construct psfs and arrays for plotting"""
# Initial PSFs
initial_psfs = model.model()

# Get plain and aberrated psfs
psf_tel = tel.set(['detector'], [None])
psf = psf_tel.set(zernikes, np.zeros(len(tel.get(zernikes)))).model(source=dl.PointSource(wavelengths=wavels))
aberrated_psf = psf_tel.model(source=dl.PointSource(wavelengths=wavels))
np.save(paths.data / "make_model_and_data/plain_psf", psf)
np.save(paths.data / "make_model_and_data/aberrated_psf", aberrated_psf)

# Save models and data
p.dump(tel, open(paths.data / "make_model_and_data/instrument.p", 'wb'))
p.dump(source, open(paths.data / "make_model_and_data/source.p", 'wb'))
np.save(paths.data / "make_model_and_data/data", data)
np.save(paths.data / "make_model_and_data/initial_psfs", initial_psfs)
p.dump(model, open(paths.data / 'make_model_and_data/model.p', 'wb'))