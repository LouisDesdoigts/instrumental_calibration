import jax.numpy as np
import jax.random as jr
from jax.scipy.stats import poisson
import equinox as eqx
import optax
import paths
import dill as p
from tqdm import tqdm

# Load model
model = p.load(open(paths.data / 'make_model_and_data/model.p', 'rb'))
data = np.load(paths.data / "make_model_and_data/data.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

# Optimisation hyper parameters
b1 = .75 # Momentum -> Higer = more momentum
# b2 = 0.5 # Acceleration -> Higer = more momentum

# Position
pos_lr, pos_start, pos_stop, pos_restart = 1e-8, 0, 50, 75
pos_sched = optax.piecewise_constant_schedule(init_value=pos_lr*1e-8,
                             boundaries_and_scales={pos_start   : int(1e8),
                                                    pos_stop    : int(1e-8),
                                                    pos_restart : int(1e8)})
pos_optimiser   = optax.adam(pos_sched, b1=b1)

# Flux
flux_lr, flux_start, flux_stop, flux_restart = 2.5e6, 0, 50, 75
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


# Combine the optimisers into a list
optimisers = [pos_optimiser, flux_optimiser, coeff_optimiser, FF_optimiser]

# Generate out optax optimiser, and also get our args
optim, opt_state, args = model.get_optimiser(parameters, optimisers, get_args=True)

def log_like(model, data):
    psfs = np.maximum(model.model(), 1e-8)
    return -np.sum(poisson.logpmf(data, psfs))

def log_prior(model, ff_mean=1., ff_std=0.05):
    return 0.5*(np.square((ff_mean - model.get(flatfield))/ff_std)).sum()

@eqx.filter_jit
@eqx.filter_value_and_grad(arg=args)
def loss_fn(model, data):
    return log_prior(model) + log_like(model, data)

# Compile
loss, grads = loss_fn(model, data)

# Optimise
losses, models_out = [], []
with tqdm(range(100), desc='Gradient Descent') as t:
    for i in t:
        loss, grads = loss_fn(model, data)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        losses.append(loss)
        models_out.append(model)
        t.set_description("Log Loss: {:.3f}".format(np.log10(loss))) # update the progress bar

# Save model and losses
np.save(paths.data / 'optimise/losses', np.array(losses))
p.dump(models_out, open(paths.data / "optimise/models_out.p", 'wb'))

# Get final PSFs
psfs_out = models_out[-1].model()
np.save(paths.data / 'optimise/final_psfs', psfs_out)

# Pre calc FF errors
thresh = 1000
fmask = data >= thresh
out_mask = np.where(data < thresh)
in_mask = np.where(data >= thresh)

data_tile = np.tile(data, [len(models_out), 1, 1])
in_mask_tiled = np.where(data_tile >= thresh)

# calculate residuals
tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
pix_response = tel.get(flatfield)
flatfields_found = np.array([model.get(flatfield) for model in models_out])
pr_residuals = pix_response[in_mask] - flatfields_found[-1][in_mask]

# for correlation plot
true_pr_masked = pix_response.at[out_mask].set(1)
found_pr_masked = flatfields_found[-1].at[out_mask].set(1)

# FF Scatter Plot
data_sum = data
colours = data_sum.flatten()
ind = np.argsort(colours)
colours = colours[ind]

pr_true_flat = true_pr_masked.flatten()
pr_found_flat = found_pr_masked.flatten()

pr_true_sort = pr_true_flat[ind]
pr_found_sort = pr_found_flat[ind]

# Errors
pfound = flatfields_found[in_mask_tiled].reshape([len(models_out), len(in_mask[0])])
ptrue = pix_response[in_mask]
pr_res = ptrue - pfound
masked_error = np.abs(pr_res).mean(-1)

# FF Scatter Plot
data_sum = data
colours = data_sum.flatten()
ind = np.argsort(colours)
colours = colours[ind]

pr_true_flat = true_pr_masked.flatten()
pr_found_flat = found_pr_masked.flatten()

pr_true_sort = pr_true_flat[ind]
pr_found_sort = pr_found_flat[ind]

np.save(paths.data / 'optimise/true_prf_sorted', pr_true_sort)
np.save(paths.data / 'optimise/found_prf_sorted', pr_found_sort)
np.save(paths.data / 'optimise/colours', colours)

# Histogram
thresh_indx = np.where(fmask)
res = (pix_response - flatfields_found[-1])[thresh_indx].flatten()
counts, bins = np.histogram(res.flatten(), bins=51)
np.save(paths.data / "optimise/pixel_response_resid_counts.npy", counts)
np.save(paths.data / "optimise/pixel_response_resid_bins.npy", bins)

# Parameters out
# positions = 'MultiPointSource.position'
# fluxes = 'MultiPointSource.flux'
# zernikes = 'ApplyBasisOPD.coefficients'
# flatfield = 'ApplyPixelResponse.pixel_response'
# parameters = [positions, fluxes, zernikes, flatfield]

# # Get parameters
positions_found  = np.array([model.get(positions) for model in models_out])
fluxes_found     = np.array([model.get(fluxes)    for model in models_out])
zernikes_found   = np.array([model.get(zernikes)  for model in models_out])
flatfields_found = np.array([model.get(flatfield) for model in models_out])

np.save(paths.data / "optimise/positions_found.npy", positions_found)
np.save(paths.data / "optimise/fluxes_found.npy", fluxes_found)
np.save(paths.data / "optimise/zernikes_found.npy", zernikes_found)
np.save(paths.data / "optimise/flatfields_found.npy", flatfields_found)
