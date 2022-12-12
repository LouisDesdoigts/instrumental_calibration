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
model = p.load(open(paths.data / 'model.p', 'rb'))
data = np.load(paths.data / "data.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

# Optimisation hyper parameters
b1 = .7 # Momentum -> Higer = more momentum
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
    psfs = np.maximum(model.observe(), 1e-8)
    return -np.sum(jax.scipy.stats.poisson.logpmf(data, psfs))

def log_prior(model, ff_mean=1., ff_std=0.05):
    return 0.5*(np.square((ff_mean - model.get(flatfield))/ff_std)).sum()

@eqx.filter_jit
@eqx.filter_value_and_grad(arg=args)
def loss_fn(model, data):
    return log_prior(model) + log_like(model, data)

# %%time
loss, grads = loss_fn(model, data) # Compile
print(loss)
print("Initial Loss: {}".format(int(loss)))
print("Initial Loss: {}".format(np.log10(loss)))


losses, models_out = [], []
with tqdm(range(100), desc='Gradient Descent') as t:
    for i in t:
        loss, grads = loss_fn(model, data)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        losses.append(loss)
        models_out.append(model)
        t.set_description("Log Loss: {:.3f}".format(np.log10(loss))) # update the progress bar

np.save(paths.data / 'losses', np.array(losses))
p.dump(models_out, open(paths.data / "models_out.p", 'wb'))