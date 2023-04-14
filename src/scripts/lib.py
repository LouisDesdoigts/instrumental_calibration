import os
import time as t
import paths
import optax
import zodiax as zdx
from tqdm import tqdm
from jax import numpy as np, random as jr
from jax.scipy.stats import poisson
from zodiax.experimental.serialisation import serialise, deserialise
from observation import MultiImage, IntegerDither
import dLux as dl
import dLux.utils as dlu
from jax import jvp, grad, jit, linearize

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def format_time(seconds):
    """Format time in seconds to hours, minutes and seconds"""
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    if minutes == 0:
        return f"{int(seconds):02d} sec"
    return f"{minutes:02d} min {int(seconds):02d} sec"

def run_script(script_name, sub_dir, make_dirs=False):
    t0 = t.time()
    print(f"Running {script_name}.py ")
    if make_dirs:
        mkdir(paths.data / script_name)
        mkdir(paths.data / script_name / sub_dir)
    os.system(f'python {script_name}.py {sub_dir}')
    print(f"Executed in {format_time(t.time() - t0)}")

def weighted_pearson_corr_coeff(x, y, weights):
    """
    Calculates the weighted Pearson correlation coefficient between two arrays of data.
    
    Parameters:
    x (numpy array): Array of data for the first variable
    y (numpy array): Array of data for the second variable
    weights (numpy array): Array of weights, one for each observation
    
    Returns:
    w (float): Weighted Pearson correlation coefficient
    """
    # Calculate the weighted mean of x and y
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    
    # Calculate the weighted covariance and standard deviations
    cov = np.sum(weights * (x - x_mean) * (y - y_mean))
    std_x = np.sqrt(np.sum(weights * (x - x_mean)**2) + 1e-12)
    std_y = np.sqrt(np.sum(weights * (y - y_mean)**2) + 1e-12)
    
    # Calculate the weighted Pearson correlation coefficient
    w = cov / (std_x * std_y)
    
    return w

def pearson_corr_coeff(x, y):
    """
    Calculates the Pearson correlation coefficient between two arrays of data.
    
    Parameters:
    x (numpy array): Array of data for the first variable
    y (numpy array): Array of data for the second variable
    
    Returns:
    r (float): Pearson correlation coefficient
    """
    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate the covariance and standard deviations
    cov = np.sum((x - x_mean) * (y - y_mean))
    std_x = np.sqrt(np.sum((x - x_mean)**2))
    std_y = np.sqrt(np.sum((y - y_mean)**2))
    
    # Calculate the Pearson correlation coefficient
    r = cov / (std_x * std_y)
    
    return r

# Likelihood
def log_like(model, data):
    return -poisson.logpmf(data, model.observe()).sum()

# Prior
def log_prior(model, PRFdev=0.01, PRF_mean=1.):
    PRF = model.get('PRF.pixel_response')
    return np.square((PRF_mean - PRF) / PRFdev).sum()

def optimise(flux, PRFdev):
    sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"
    mkdir(paths.data / sub_dir)

    # Wavefront Parameters
    aperture = 0.5
    wf_npix = 512
    det_npix = 1024
    sampling_rate = 20
    det_dev = 0
    det_mean = 0
    xy_max = 4.5
    Nstars = 20
    Nims = 1

    ### Construct Optics ###
    wavels = 1e-9 * np.linspace(545, 645, 3)
    zernike_inds = np.arange(4, 11)
    coeffs = 1e-8 * jr.normal(jr.PRNGKey(0), [len(zernike_inds)])

    det_pixsize = dlu.get_pixel_scale(sampling_rate, wavels.mean(), aperture)
    raw_mask = dlu.phase_to_opd(np.load(paths.scripts / "mask.npy"), wavels.mean())
    mask = dlu.scale_array(raw_mask, wf_npix, 0)

    optics = dl.Optics([
        dl.CreateWavefront    (wf_npix, aperture),
        dl.ApertureFactory    (wf_npix, secondary_ratio=0.2),
        dl.AberrationFactory  (wf_npix, zernikes=zernike_inds, coefficients=coeffs,
                            name='Aberrations'),
        dl.AddOPD             (mask),
        dl.NormaliseWavefront (),
        dl.AngularMFT         (det_npix, det_pixsize)])

    ### Detector ###
    pix_response = 1 + PRFdev*jr.normal(jr.PRNGKey(1), [det_npix, det_npix])
    detector = dl.Detector([
        dl.ApplyPixelResponse(pix_response, name='PRF'),
        dl.AddConstant(det_mean)])

    ### Source ###
    true_fluxes = flux + (flux/10)*jr.normal(jr.PRNGKey(2), (Nstars,))
    true_positions = dlu.s2r(jr.uniform(jr.PRNGKey(4), (Nstars, 2), 
        minval=-xy_max, maxval=xy_max))
    source = dl.MultiPointSource(true_positions, true_fluxes, 
        wavelengths=wavels, name='Source')

    ### Instrument ###
    observation = IntegerDither([[0, 0], [20, 20], [20, -20], [-20, 20], [-20, -20]])
    tel = dl.Instrument(optics, source, detector, observation)
    serialise(paths.data / f'{sub_dir}/instrument.zdx', tel)

    ### Make Data ###
    bg_val = tel.AddConstant.value
    photons = jr.poisson(jr.PRNGKey(2), tel.observe())
    BG_noise = det_dev*jr.normal(jr.PRNGKey(3), photons.shape) + bg_val
    data = photons + BG_noise
    np.save(paths.data / f"{sub_dir}/data", data)

    ### Initial Model ###
    model = tel.add('Source.position', 
        1.*det_pixsize*jr.normal(jr.PRNGKey(6),  (Nstars, 2)))
    model = model.multiply('Source.flux', 
        1 + 0.1*jr.normal(jr.PRNGKey(7), (Nstars,)))
    model = model.set('Aberrations.coefficients',
        np.zeros(len(zernike_inds)))
    model = model.set('PRF.pixel_response', 
        np.ones((det_npix, det_npix)))
    
    serialise(paths.data / f"{sub_dir}/initial_model", model)

    
    ### Optimisation ###
    b1 = .75 # Momentum
    pos_lr = 1e-8
    coeff_lr = 1e-9
    flux_lr = flux * 1e-2
    PRF_lr = 2e-1
    PRF_lr *= PRFdev

    pos_optimiser   = optax.adam(pos_lr,    b1=b1)
    flux_optimiser  = optax.adam(flux_lr,   b1=b1)
    coeff_optimiser = optax.adam(coeff_lr,  b1=b1)

    parameters1 = ['Source.position', 'Source.flux', 'Aberrations.coefficients']
    optimisers1 = [pos_optimiser, flux_optimiser, coeff_optimiser]

    # Loss1
    @zdx.filter_jit
    @zdx.filter_value_and_grad(parameters1)
    def loss_fn1(model, data, PRFdev=0.01):
        return log_like(model, data) + log_prior(model, PRFdev)

    # Compile
    loss, grads = loss_fn1(model, data)
    print(f"Initial Log10 Loss: {np.log10(loss):.5}")

    ### Optimise Params ###
    losses, models_out = [], []
    optim1, opt_state1 = zdx.get_optimiser(model, parameters1, optimisers1)
    with tqdm(range(100), desc='Gradient Descent') as t:
        for i in t:
            loss, grads = loss_fn1(model, data, PRFdev)
            updates, opt_state1 = optim1.update(grads, opt_state1)
            model = zdx.apply_updates(model, updates)
            losses.append(loss)
            models_out.append(model)
            t.set_description("Log10 Loss: {:.3f}".format(np.log10(loss)))

    # Second optimisation
    parameters2 = ['PRF.pixel_response']
    PRF_optimiser = optax.adam(PRF_lr, b1=b1)
    optimisers2 = [PRF_optimiser]

    # Loss2
    @zdx.filter_jit
    @zdx.filter_value_and_grad(parameters2)
    def loss_fn2(model, data, PRFdev=0.01):
        return log_like(model, data) + log_prior(model, PRFdev)
    
    # Compile
    loss, grads = loss_fn2(model, data)

    ### Optimise PRF ###
    optim2, opt_state2 = zdx.get_optimiser(model, parameters2, optimisers2)
    with tqdm(range(50), desc='Gradient Descent') as t:
        for i in t:
            loss, grads = loss_fn2(model, data, PRFdev)
            updates, opt_state2 = optim2.update(grads, opt_state2)
            model = zdx.apply_updates(model, updates)
            losses.append(loss)
            models_out.append(model)
            t.set_description("Log10 Loss: {:.3f}".format(np.log10(loss)))

    ### Save ###
    serialise(paths.data / sub_dir / "final_model.zdx", model)

def calc_errors(flux, PRFdev):
    sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

    # Load model
    tel = deserialise(paths.data / f'{sub_dir}/instrument.zdx')
    final_model = deserialise(paths.data / f'{sub_dir}/final_model.zdx')
    data = np.load(paths.data / f"{sub_dir}/data.npy")

    positions = 'Source.position'
    fluxes = 'Source.flux'
    zernikes = 'Aberrations.coefficients'

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

    np.save(paths.data / f'{sub_dir}/cov_mat', cov_mat)