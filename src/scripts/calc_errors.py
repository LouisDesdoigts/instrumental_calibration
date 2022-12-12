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
import dill as p

# Plotting/visualisation
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Load model
tel = p.load(open(paths.data / 'instrument.p', 'rb'))
models_out = p.load(open(paths.data / 'models_out.p', 'rb'))
losses = np.load(paths.data / 'losses.npy')
data = np.load(paths.data / "data.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'
parameters = [positions, fluxes, zernikes, flatfield]

def perturb(X, model):
    """
    Perturbs the values of the model
    """
    model = model.add(positions, X[:2*Nstars].reshape((Nstars, 2)))
    model = model.add(fluxes, X[2*Nstars:3*Nstars])
    model = model.add(zernikes, X[3*Nstars:])
    return model

Nstars = len(tel.get(positions))
Nzern  = len(tel.get(zernikes))
X = np.zeros(3*Nstars + Nzern)

from jax import jvp, grad, jit, linearize

def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

def hessian(f, x):
    _, hvp = linearize(jax.grad(f), x)
    hvp = jit(hvp)  # seems like a substantial speedup to do this
    basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

model = tel.set(['observation'], [None])
fn = lambda X: dl.utils.poisson_log_likelihood(X, data[0], model, perturb, 'model')

H = hessian(fn, X)
cov_mat = -np.linalg.inv(H)

np.save('cov_mat', cov_mat)



# from dLux.utils import bayes
# cov_mat = bayes.calculate_covariance(bayes.poisson_log_likelihood, X, data, models_out[-1], perturb, 'model')

# # print(np.diag(cov_mat)**0.5)
# # np.save('cov_mat', cov_mat)
