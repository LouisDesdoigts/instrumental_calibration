import jax.numpy as np
from jax import jvp, grad, jit, linearize
import dLux as dl
import paths
import pickle as p
from tqdm import tqdm

import os
try:
    os.mkdir(paths.data / "calc_errors")
except FileExistsError:
    pass

# Load model
tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
final_model = p.load(open(paths.data / 'optimise/final_model.p', 'rb'))
data = np.load(paths.data / "make_model_and_data/data.npy")

positions = 'MultiPointSource.position'
fluxes = 'MultiPointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'

def perturb(X, model):
    """
    Perturbs the values of the model
    """
    model = model.add(positions, X[:2*Nstars].reshape((Nstars, 2)))
    model = model.add(fluxes, X[2*Nstars:3*Nstars])
    model = model.add(zernikes, X[3*Nstars:])
    return model

def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

def hessian(f, x):
    _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)  # seems like a substantial speedup to do this
    basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

Nstars = len(tel.get(positions))
Nzern  = len(tel.get(zernikes))
X = np.zeros(3*Nstars + Nzern)

model = final_model
fn = lambda X: dl.utils.poisson_log_likelihood(X, data, model, perturb, 'model')
cov_mat = -np.linalg.inv(hessian(fn, X))

np.save(paths.data / 'calc_errors/cov_mat', cov_mat)