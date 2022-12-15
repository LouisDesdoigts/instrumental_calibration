import jax.numpy as np
import jax.random as jr
from jax.scipy.stats import poisson
import equinox as eqx
import optax
import dLux as dl
import paths
import dill as p
from tqdm import tqdm

import os
try:
    os.mkdir(paths.data / "divergence")
except FileExistsError:
    pass

# Define wavelengths
wavels = 1e-9 * np.linspace(545, 645, 3)

# Basic Optical Parameters
aperture = 0.5
wf_npix = 512

# Detector Parameters
det_npix = 100
sampling_rate = 3
det_pixsize = dl.utils.get_pixel_scale(sampling_rate, wavels.mean(), aperture)

# Load mask
# raw_mask = dl.utils.phase_to_opd(np.load("mask.npy"), wavels.mean())
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
pix_response = 1 + 0.05*jr.normal(jr.PRNGKey(0), [det_npix, det_npix])

# Create Detector object
detector = dl.Detector([dl.ApplyPixelResponse(pix_response), dl.AddConstant(10.)])

# Create Source object
source = dl.PointSource(np.zeros(2), 1e3, wavelengths=wavels)

# # Observation stratergy, nims
# nims = 1
# def observe_fn(model, nims):
#     psf = model.apply('MultiPointSource.flux', lambda x: x/nims).model()
#     return np.tile(psf, (nims, 1, 1))
# observation = {'fn': observe_fn, 'args': nims}

# Combine into instrument
tel = dl.Instrument(optics=optics, sources=[source], detector=detector)


positions = 'PointSource.position'
fluxes = 'PointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'

parameters = [positions, fluxes, zernikes, flatfield]


@eqx.filter_vmap(args=(None, 0))
def make_instruments(model, flux):
    return model.set('PointSource.flux', np.array(flux))

@eqx.filter_vmap
def evaluate_ensemble(model):
    # return model.observe()
    return model.model()

@eqx.filter_vmap
def make_images(model, key):
    bg_val = model.AddConstant.value
    # psfs = model.observe() - bg_val
    # psfs = model.set(['detector'], [None]).observe()
    psfs = model.set(['detector'], [None]).model()
    BG_noise = 2.5*jr.normal(jr.PRNGKey(key), psfs.shape) + bg_val
    data = pix_response*jr.poisson(jr.PRNGKey(key), psfs) + BG_noise
    return data

@eqx.filter_vmap
def initialise_models(model):
    model = model.add(positions, 1.*det_pixsize*jr.normal(jr.PRNGKey(0),  (2,)))
    # model = model.add(positions, 1.*det_pixsize)

    # Multiply the fluxes by small random values
    model = model.multiply(fluxes, 1 + 0.1*jr.normal(jr.PRNGKey(1), (1,)))

    # Set the zernike coefficients to zero
    model = model.set(zernikes, np.zeros(len(zern_basis)))

    # Set the flat field to uniform
    model = model.set(flatfield, np.ones((det_npix, det_npix)))

    return model

fluxes_in = 10**np.linspace(3, 6, 20)
np.save(paths.data / "divergence/divergence_fluxes_in", fluxes_in)
flux_ratios = fluxes_in/fluxes_in[0]
tels = make_instruments(tel, fluxes_in)
psfs = evaluate_ensemble(tels)
data = make_images(tels, np.ones(len(fluxes_in), dtype=int))
models = initialise_models(tels)



# Optimisation hyper parameters
b1 = .75 # Momentum -> Higer = more momentum
# b2 = 0.5 # Acceleration -> Higer = more momentum

# Position
pos_lr, pos_start, pos_stop, pos_restart = 1e-8, 0, 50, 75
pos_lr, pos_start, pos_stop, pos_restart = 5e-8, 0, 50, 75
pos_sched = optax.piecewise_constant_schedule(init_value=pos_lr*1e-8,
                             boundaries_and_scales={pos_start   : int(1e8),
                                                    pos_stop    : int(1e-8),
                                                    pos_restart : int(1e8)})
pos_optimiser   = optax.adam(pos_sched, b1=b1)

# Flux
flux_lr = 0.5e3*flux_ratios[:, None] * np.ones(models.get(fluxes).shape)
flux_start, flux_stop, flux_restart = 0, 50, 75
flux_sched = optax.piecewise_constant_schedule(init_value=flux_lr*1e-8,
                             boundaries_and_scales={flux_start   : int(1e8),
                                                    flux_stop    : int(1e-8),
                                                    flux_restart : int(1e8)})
flux_optimiser = optax.adam(flux_sched, b1=b1)

# Zernikes
coeff_lr, coeff_start, coeff_stop, coeff_restart = 5e-9, 0, 50, 75
coeff_sched = optax.piecewise_constant_schedule(init_value=coeff_lr*1e-8,
                             boundaries_and_scales={coeff_start   : int(1e8),
                                                    coeff_stop    : int(1e-8),
                                                    coeff_restart : int(1e8)})
coeff_optimiser = optax.adam(coeff_sched, b1=b1)

# FF
ff_lr, ff_start = 1e-2, 50
FF_sched = optax.piecewise_constant_schedule(init_value=ff_lr*1e-8,
                             boundaries_and_scales={ff_start : int(1e8)})
FF_optimiser = optax.adam(FF_sched, b1=b1)

optimisers = [pos_optimiser, flux_optimiser, coeff_optimiser, FF_optimiser]
optim, opt_state, args = models.get_optimiser(parameters, optimisers, get_args=True)

def log_prior(model, ff_mean=1., ff_std=0.05):
    return 0.5*(np.square((ff_mean - model.get(flatfield))/ff_std)).sum()

def log_like(model, data):
    psfs = np.maximum(model.model(), 1e-8)
    return -np.sum(poisson.logpmf(data, psfs))

@eqx.filter_vmap
@eqx.filter_jit
@eqx.filter_value_and_grad(arg=args)
def loss_fn(model, data):
    return log_prior(model) + log_like(model, data)

# Compile
losses, grads = loss_fn(models, data) 

# Optimise
losses_out, models_out = [], []
with tqdm(range(100),desc='Gradient Descent') as t:
    for i in t:
        losses, grads = loss_fn(models, data)
        update, opt_state = optim.update(grads, opt_state)
        models = eqx.apply_updates(models, update)
        losses_out.append(losses)
        models_out.append(models)
        t.set_description("Log Loss: {:.3f}".format(np.log10(losses).mean())) # update the progress bar

losses_out = np.array(losses_out)

# Save models
# p.dump(models_out, open(paths.data / 'divergence/divergence_models_out.p', 'wb'))

positions_found  = np.array([model.get(positions) for model in models_out])
fluxes_found = np.array([model.get(fluxes) for model in models_out])

np.save(paths.data / "divergence/positions_found.npy", positions_found)
np.save(paths.data / "divergence/fluxes_found.npy", fluxes_found)