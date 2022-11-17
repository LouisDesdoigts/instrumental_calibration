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
# source = p.load(open(paths.data / 'source.p', 'rb'))
data = np.load(paths.data / "data.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

# So first we simply set the simple parameters to use an adam optimiser 
# algorithm, with individual learning rates
pos_optimiser   = optax.adam(2e-8)
flux_optimiser  = optax.adam(1e6)
coeff_optimiser = optax.adam(2e-9)

# Now the flat-field, becuase it is highly covariant with the mean flux level 
# we don't start learning its parameters until the 100th epoch.
FF_sched = optax.piecewise_constant_schedule(init_value=1e-2*1e-8, 
                             boundaries_and_scales={100 : int(1e8)})
FF_optimiser = optax.adam(FF_sched)

# Combine the optimisers into a list
optimisers = [pos_optimiser, flux_optimiser, coeff_optimiser, FF_optimiser]

# Generate out optax optimiser, and also get our args
optim, opt_state, args = model.get_optimiser(parameters, optimisers, get_args=True)


@eqx.filter_jit
@eqx.filter_value_and_grad(arg=args)
def loss_fn(model, data):
    out = model.observe()
    return -np.sum(jax.scipy.stats.poisson.logpmf(data, out))


# %%time
loss, grads = loss_fn(model, data) # Compile
print("Initial Loss: {}".format(int(loss)))


# %%timeit
# loss = loss_fn(model, data)[0].block_until_ready() # Compile



losses, models_out = [], []
with tqdm(range(200),desc='Gradient Descent') as t:
    for i in t:
        loss, grads = loss_fn(model, data)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        losses.append(loss)
        models_out.append(model)
        t.set_description("Log Loss: {:.3f}".format(np.log10(loss))) # update the progress bar

np.save(paths.data /'losses', np.array(losses))
p.dump(models_out, open(paths.data / "models_out.p", 'wb'))