import os
import time as t
import paths
import jax.numpy as np
import jax.random as jr
from jax.scipy.stats import poisson
import equinox as eqx
import optax
import paths
from tqdm import tqdm
import zodiax as zdx
from zodiax.experimental.serialisation import serialise, deserialise
from observation import MultiImage, IntegerDither
import dLux.utils as dlu
import dLux as dl
# from lib import mkdir
import sys

# Core jax
import jax.numpy as np
import jax.random as jr
import dLux as dl
import dLux.utils as dlu
import paths
import json
import time
import os
import sys
from zodiax.experimental.serialisation import serialise

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

def make_model_and_data(flux, PRFdev):
    sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

    mkdir(paths.data / "make_model_and_data")
    mkdir(paths.data / "make_model_and_data/" / sub_dir)
    mkdir(paths.figures / sub_dir)


    pos_lr = 1e-8 # Normal
    coeff_lr = 1e-9
    flux_lr = flux * 1e-2

    PRF_lr = 2e-1
    PRF_lr *= PRFdev
    PRF_lr /= 1

    params = {
        "Flux"       : np.array(flux),
        "PRFdev"     : np.array(PRFdev),
        "pos_lr"     : pos_lr,
        "flux_lr"    : flux_lr,
        "coeff_lr"   : coeff_lr,
        "PRF_lr"     : PRF_lr,
        "PRF_start"  : 75,
        "transition" : 100,
        "Momentum"   : .75,
    }

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
    raw_mask = dlu.phase_to_opd(np.load(paths.data / "mask.npy"), wavels.mean())
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
    tel = dl.Instrument(optics=optics, sources=[source], detector=detector, observation=observation)

    ### Make Data ###
    bg_val = tel.AddConstant.value
    photons = jr.poisson(jr.PRNGKey(2), tel.observe())
    BG_noise = det_dev*jr.normal(jr.PRNGKey(3), photons.shape) + bg_val
    data = photons + BG_noise

    ### Initial Model ###
    model = tel.add('Source.position', 
        1.*det_pixsize*jr.normal(jr.PRNGKey(6),  (Nstars, 2)))
    model = model.multiply('Source.flux', 
        1 + 0.1*jr.normal(jr.PRNGKey(7), (Nstars,)))
    model = model.set('Aberrations.coefficients',
        np.zeros(len(zernike_inds)))
    model = model.set('PRF.pixel_response', 
        np.ones((det_npix, det_npix)))

    # Save
    with open(paths.figures / f"{sub_dir}/params.txt", "w") as file:
        file.write(f"Date: {time.strftime('%Y-%m-%d', time.localtime())}\n")
        file.write(f"Time: {time.strftime('%H:%M:%S', time.localtime())}\n")

        # Sim params
        file.write("\nSimulation Parameters\n")
        file.write(f"  Nstars:   {Nstars}\n")
        file.write(f"  Flux:     {flux:.0e}\n")
        file.write(f"  PRFdev:   {PRFdev:.0e}\n")
        file.write(f"  det_mean: {det_mean}\n")
        file.write(f"  det_dev:  {det_dev}\n")

        # Optimisation params
        file.write("\nOptimisation Parameters\n")
        file.write(f"  Momentum:   {params['Momentum']}\n")
        file.write(f"  pos_lr:     {pos_lr:.0e}\n")
        file.write(f"  coeff_lr:   {coeff_lr:.0e}\n")
        file.write(f"  flux_lr:    {flux_lr:.0e}\n")
        file.write(f"  PRF_lr:     {PRF_lr:.0e}\n")
        file.write(f"  PRF_start:  {params['PRF_start']}\n")
        file.write(f"  transition: {params['transition']}\n")

    serialise(paths.data / "make_model_and_data/" / sub_dir / "params.zdx", params)
    serialise(paths.data / f'make_model_and_data/{sub_dir}/instrument.zdx', tel)
    serialise(paths.data / f'make_model_and_data/{sub_dir}/model.zdx', model)
    np.save(paths.data / f"make_model_and_data/{sub_dir}/BG_noise", BG_noise)
    np.save(paths.data / f"make_model_and_data/{sub_dir}/data", data)


def optimise(flux, PRFdev):
    sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"
    
    mkdir(paths.data / "optimise")
    mkdir(paths.data / "optimise" / sub_dir)

    params = deserialise(paths.data/"make_model_and_data"/sub_dir/"params.zdx")


    b1 = params['Momentum']
    pos_optimiser   = optax.adam(params['pos_lr'],    b1=b1)
    flux_optimiser  = optax.adam(params['flux_lr'],   b1=b1)
    coeff_optimiser = optax.adam(params['coeff_lr'],  b1=b1)

    parameters1 = ['Source.position', 'Source.flux', 'Aberrations.coefficients']
    optimisers1 = [pos_optimiser, flux_optimiser, coeff_optimiser]


    PRF_optimiser = optax.adam(params['PRF_lr'], b1=b1)

    parameters2 = ['PRF.pixel_response']
    optimisers2 = [PRF_optimiser]

    # Likelihood
    def log_like(model, data):
        return -poisson.logpmf(data, model.observe()).sum()

    # Prior
    def log_prior(model, PRFdev=0.01, PRF_mean=1.):
        PRF = model.get('PRF.pixel_response')
        return np.square((PRF_mean - PRF) / PRFdev).sum()

    # Loss1
    @zdx.filter_jit
    @zdx.filter_value_and_grad(parameters1)
    def loss_fn1(model, data, PRFdev=0.01):
        return log_like(model, data) + log_prior(model, PRFdev)

    # Loss2
    @zdx.filter_jit
    @zdx.filter_value_and_grad(parameters2)
    def loss_fn2(model, data, PRFdev=0.01):
        return log_like(model, data) + log_prior(model, PRFdev)

    # Compile
    model = deserialise(paths.data / "make_model_and_data" / sub_dir / "model.zdx")
    data = np.load(paths.data / "make_model_and_data" / sub_dir / "data.npy")
    loss, grads = loss_fn1(model, data)
    print(f"Initial Log10 Loss: {np.log10(loss):.5}")
    loss, grads = loss_fn2(model, data)

    ### Optimise Params ###
    losses, models_out = [], []
    optim1, opt_state1 = zdx.get_optimiser(model, parameters1, optimisers1)
    # with tqdm(range(100), desc='Gradient Descent') as t:
    with tqdm(range(1), desc='Gradient Descent') as t:
        for i in t:
            loss, grads = loss_fn1(model, data, params['PRFdev'])
            updates, opt_state1 = optim1.update(grads, opt_state1)
            model = zdx.apply_updates(model, updates)
            losses.append(loss)
            models_out.append(model)
            t.set_description("Log10 Loss: {:.3f}".format(np.log10(loss)))

    ### Optimise PRF ###
    optim2, opt_state2 = zdx.get_optimiser(model, parameters2, optimisers2)
    # with tqdm(range(100), desc='Gradient Descent') as t:
    # with tqdm(range(50), desc='Gradient Descent') as t:
    with tqdm(range(1), desc='Gradient Descent') as t:
        for i in t:
            loss, grads = loss_fn2(model, data, params['PRFdev'])
            updates, opt_state2 = optim2.update(grads, opt_state2)
            model = zdx.apply_updates(model, updates)
            losses.append(loss)
            models_out.append(model)
            t.set_description("Log10 Loss: {:.3f}".format(np.log10(loss)))

    # ### Save ###
    serialise(paths.data / "optimise" / sub_dir / "final_model.zdx", models_out[-1])

def process(flux, PRFdev):
    sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

    mkdir(paths.data / "process")
    mkdir(paths.data / "process" / sub_dir)

    ### Load Data ###
    params = deserialise(paths.data / f"make_model_and_data/{sub_dir}/params.zdx")
    tel = deserialise(paths.data / f"make_model_and_data/{sub_dir}/instrument.zdx")
    model = deserialise(paths.data / f"make_model_and_data/{sub_dir}/model.zdx")
    final_model = deserialise(paths.data / f"optimise/{sub_dir}/final_model.zdx")
    data = np.load(paths.data / f"make_model_and_data/{sub_dir}/data.npy")

    ### Parameters ###
    positions  = 'Source.position'
    fluxes     = 'Source.flux'
    zernikes   = 'Aberrations.coefficients'
    flatfield  = 'PRF.pixel_response'

    ### Expected Errors ###
    mean_wl = tel.Source.spectrum.wavelengths.mean()
    lamd = mean_wl/tel.CreateWavefront.diameter
    true_fluxes = tel.get(fluxes)
    expected_flux_error = np.mean(np.sqrt(true_fluxes)) # Photons
    expected_postional_error = 1/np.pi * np.sqrt(1/true_fluxes) * lamd # Radians
    expected_postional_error = np.tile(expected_postional_error, (2,1)).flatten()
    expected_zern_err = 1/(np.sqrt(true_fluxes).mean()) # Radians
    np.save(paths.data / f"process/{sub_dir}/expected_flux_error.npy", expected_flux_error)
    np.save(paths.data / f"process/{sub_dir}/expected_postional_error.npy", expected_postional_error)
    np.save(paths.data / f"process/{sub_dir}/expected_zern_err.npy", expected_zern_err)

    ### Full PSFs ###
    true_psfs = tel.observe()
    initial_psfs = model.observe()
    final_psfs = final_model.observe()
    np.save(paths.data / f"process/{sub_dir}/true_psfs", true_psfs)
    np.save(paths.data / f"process/{sub_dir}/initial_psfs", initial_psfs)
    np.save(paths.data / f'process/{sub_dir}/final_psfs', final_psfs)

    ### Residual Histogram ###
    res_norm = (data - final_psfs)/true_psfs ** 0.5
    counts, bins = np.histogram(res_norm.flatten(), bins=101)
    np.save(paths.data / f"process/{sub_dir}/res_norm_counts", counts)
    np.save(paths.data / f'process/{sub_dir}/res_norm_bins', bins)

    ### Plain PSF ###
    optics = tel.get('optics')
    wavels = tel.get('Source.spectrum.wavelengths')
    unaberrated_optics = optics.multiply(zernikes, 0.)
    unit_psf = unaberrated_optics.propagate(wavels)
    np.save(paths.data / f"process/{sub_dir}/unit_psf", unit_psf)

    ### Aberrated PSF ###
    aberrated_psf = optics.model(dl.PointSource(wavelengths=wavels))
    np.save(paths.data / f"process/{sub_dir}/aberrated_psf", aberrated_psf)

    ### PRF ###
    PRF = tel.get(flatfield)
    counts, bins = np.histogram(PRF.flatten(), bins=51)
    np.save(paths.data / f"process/{sub_dir}/PRdev.npy", params['PRFdev'])
    np.save(paths.data / f"process/{sub_dir}/PRF.npy", PRF)
    np.save(paths.data / f"process/{sub_dir}/PRF_counts.npy", counts)
    np.save(paths.data / f"process/{sub_dir}/PRF_bins.npy", bins)

    # FF Scatter Plot
    data_sum = data.sum(0)
    colours = data_sum.flatten()
    ind = np.argsort(colours)
    colours = colours[ind]

    PRF_true_flat = tel.get(flatfield).flatten()
    PRF_found_flat = final_model.get(flatfield).flatten()

    PRF_true_sort = PRF_true_flat[ind]
    PRF_found_sort = PRF_found_flat[ind]

    np.save(paths.data / f'process/{sub_dir}/true_PRF_sorted', PRF_true_sort)
    np.save(paths.data / f'process/{sub_dir}/found_PRF_sorted', PRF_found_sort)
    np.save(paths.data / f'process/{sub_dir}/colours', colours)

    # Recovered PRF Histogram
    found_PRF = final_model.get(flatfield)
    counts, bins = np.histogram(found_PRF.flatten(), bins=51)
    np.save(paths.data / f"process/{sub_dir}/PRF_found_counts.npy", counts)
    np.save(paths.data / f"process/{sub_dir}/PRF_found_bins.npy", bins)

    # PRF Residual Histogram
    PRF_res = PRF - found_PRF
    counts, bins = np.histogram(PRF_res.flatten(), bins=51)
    np.save(paths.data / f"process/{sub_dir}/PRF_resid_counts.npy", counts)
    np.save(paths.data / f"process/{sub_dir}/PRF_resid_bins.npy", bins)

def calc_errors(flux, PRFdev):
    sub_dir = f"flux_{flux:.0e}_PRFdev_{PRFdev:.0e}"

    mkdir(paths.data / "calc_errors")
    mkdir(paths.data / "calc_errors" / sub_dir)

    # Load model
    tel = deserialise(paths.data / f'make_model_and_data/{sub_dir}/instrument.zdx')
    final_model = deserialise(paths.data / f'optimise/{sub_dir}/final_model.zdx')
    data = np.load(paths.data / f"make_model_and_data/{sub_dir}/data.npy")

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

    np.save(paths.data / f'calc_errors/{sub_dir}/cov_mat', cov_mat)