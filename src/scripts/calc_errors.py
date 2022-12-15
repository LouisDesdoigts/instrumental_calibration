import jax.numpy as np
import jax.random as jr
from jax import jvp, grad, jit, linearize
import dLux as dl
import paths
import dill as p
from tqdm import tqdm

import os
try:
    os.mkdir(paths.data / "calc_errors")
except FileExistsError:
    pass

# Load model
tel = p.load(open(paths.data / 'make_model_and_data/instrument.p', 'rb'))
# models_out = p.load(open(paths.data / 'optimise/models_out.p', 'rb'))
final_model = p.load(open(paths.data / 'optimise/final_model.p', 'rb'))
# losses = np.load(paths.data / 'optimise/losses.npy')
data = np.load(paths.data / "make_model_and_data/data.npy")

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


def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

def hessian(f, x):
    _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)  # seems like a substantial speedup to do this
    basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

# model = tel.set(['observation'], [None])
# model = tel
model = final_model
fn = lambda X: dl.utils.poisson_log_likelihood(X, data, model, perturb, 'model')

H = hessian(fn, X)
cov_mat = -np.linalg.inv(H)

np.save(paths.data / 'calc_errors/cov_mat', cov_mat)



# from dLux.utils import bayes
# cov_mat = bayes.calculate_covariance(bayes.poisson_log_likelihood, X, data, models_out[-1], perturb, 'model')

# # print(np.diag(cov_mat)**0.5)
# # np.save('cov_mat', cov_mat)
