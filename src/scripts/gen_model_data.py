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
import pickle as p

# Plotting/visualisation
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120




'''Create Optics, Detector, Source & Instrument'''
# Define wavelengths
wavels = 1e-9 * np.linspace(545, 645, 3)

# Basic Optical Parameters
aperture = 0.5
wf_npix = 512

# Detector Parameters
det_npix = 1024
sampling_rate = 20
det_pixsize = dl.utils.get_pixel_scale(sampling_rate, wavels.mean(), aperture)

# Load mask
raw_mask = dl.utils.phase_to_opd(np.load("mask.npy"), wavels.mean())
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
detector = dl.Detector([dl.ApplyPixelResponse(pix_response)])

# Observation stratergy, define dithers
dithers = 2**-.5 * det_pixsize * np.array([[+1, +1],
                                           [+1, -1],
                                           [-1, +1],
                                           [-1, -1]])

def observe_fn(model, dithers):
    return model.dither_and_model(dithers)

# Observation dictionary
observation = {'fn': observe_fn, 'args': dithers}

# Multiple sources to observe
Nstars = 20
true_positions = a2r(jr.uniform(jr.PRNGKey(0), (Nstars, 2), minval=-5, maxval=5))
true_fluxes = 1e8 + 1e7*jr.normal(jr.PRNGKey(0), (Nstars,))

# Create Source object
source = dl.MultiPointSource(true_positions, true_fluxes, wavelengths=wavels)

# Combine into instrument
tel = dl.Instrument(optics=optics, sources=[source], detector=detector,
                    observation=observation)

# Observe!
psfs = tel.observe()



'''Figure 1'''
from matplotlib.cm import get_cmap
opd = tel.ApplyBasisOPD.get_total_opd()
psf = tel.model(source=dl.PointSource(wavelengths=wavels))
throughput = tel.CompoundAperture.get_aperture(npixels=wf_npix)
pupil = opd.at[np.where(throughput==0.)].set(np.nan)

FF = tel.ApplyPixelResponse.pixel_response

cmap = get_cmap("inferno")
cmap.set_bad('k',1.)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.title("Pupil OPD")
plt.imshow(pupil * 1e9, cmap=cmap)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar()
cbar.set_label("OPD (nm)")

plt.subplot(1, 3, 2)
plt.title("PSF")
plt.imshow(psf)
plt.xlabel("Pixels")
plt.ylabel("Pixels")
cbar = plt.colorbar()
cbar.set_label("Probability")

plt.subplot(1, 3, 3)
plt.title("Pixel Response")
plt.hist(FF.flatten(), bins=25)
plt.ylabel("Counts")
plt.xlabel("Relative Sensitivity")

plt.tight_layout()
plt.savefig(paths.figures / "optics.pdf", dpi=300)
plt.show()





'''Figure 2'''
# Apply some noise to the PSF Background noise
BG_noise = np.abs(5*jr.normal(jr.PRNGKey(0), psfs.shape))
data = jr.poisson(jr.PRNGKey(0), psfs) + BG_noise

plt.figure(figsize=(20, 4))
plt.suptitle("Data")
for i in range(len(psfs)):
    plt.subplot(1, 4, i+1)
    plt.imshow(data[i]*1e-3)
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")
    cbar = plt.colorbar()
    cbar.set_label("Counts $x10^3$")
plt.tight_layout()

plt.savefig(paths.figures / "data.pdf", dpi=300)
plt.show()






# positions = 'MultiPointSource.position'
# fluxes = 'MultiPointSource.flux'
# zernikes = 'ApplyBasisOPD.coefficients'
# flatfield = 'ApplyPixelResponse.pixel_response'

# parameters = [positions, fluxes, zernikes, flatfield]



# '''Perturb'''
# # Add small random values to the positions
# model = tel.add(positions, 2.5*det_pixsize*jr.normal(jr.PRNGKey(0),  (Nstars, 2)))

# # Multiply the fluxes by small random values
# model = model.multiply(fluxes, 1 + 0.1*jr.normal(jr.PRNGKey(0), (Nstars,)))

# # Set the zernike coefficients to zero
# model = model.set(zernikes, np.zeros(len(zern_basis)))

# # Set the flat fiel to uniform
# model = model.set(flatfield, np.ones((det_npix, det_npix)))




# '''Figure 3'''
# initital_psfs = model.observe()

# plt.figure(figsize=(20, 4))
# plt.suptitle("Initial Residuals")
# for i in range(len(initital_psfs)):
#     plt.subplot(1, 4, i+1)
#     plt.imshow((initital_psfs[i] - data[i])*1e-3)
#     plt.xlabel("Pixels")
#     plt.ylabel("Pixels")
#     cbar = plt.colorbar()
#     cbar.set_label("Counts $x10^3$")
# plt.tight_layout()
# plt.show()





# '''Set up optimisation'''
# # So first we simply set the simple parameters to use an adam optimiser 
# # algorithm, with individual learning rates
# pos_optimiser   = optax.adam(2e-8)
# flux_optimiser  = optax.adam(1e6)
# coeff_optimiser = optax.adam(2e-9)

# # Now the flat-field, becuase it is highly covariant with the mean flux level 
# # we don't start learning its parameters until the 100th epoch.
# FF_sched = optax.piecewise_constant_schedule(init_value=1e-2*1e-8, 
#                              boundaries_and_scales={100 : int(1e8)})
# FF_optimiser = optax.adam(FF_sched)

# # Combine the optimisers into a list
# optimisers = [pos_optimiser, flux_optimiser, coeff_optimiser, FF_optimiser]

# # Generate out optax optimiser, and also get our args
# optim, opt_state, args = model.get_optimiser(parameters, optimisers, get_args=True)

# @eqx.filter_jit
# @eqx.filter_value_and_grad(arg=args)
# def loss_fn(model, data):
#     out = model.observe()
#     return -np.sum(jax.scipy.stats.poisson.logpmf(data, out))

# # Compile
# loss, grads = loss_fn(model, data)


# '''Optimise'''
# losses, models_out = [], []
# with tqdm(range(200),desc='Gradient Descent') as t:
#     for i in t:
#         loss, grads = loss_fn(model, data)
#         updates, opt_state = optim.update(grads, opt_state)
#         model = eqx.apply_updates(model, updates)
#         losses.append(loss)
#         models_out.append(model)
#         t.set_description("Log Loss: {:.3f}".format(np.log10(loss))) # update the progress bar


# '''Examine Results'''
# nepochs = len(models_out)
# psfs_out = models_out[-1].observe()

# positions_found  = np.array([model.get(positions) for model in models_out])
# fluxes_found     = np.array([model.get(fluxes)    for model in models_out])
# zernikes_found   = np.array([model.get(zernikes)  for model in models_out])
# flatfields_found = np.array([model.get(flatfield) for model in models_out])

# coeff_residuals = coeffs - zernikes_found
# flux_residuals = true_fluxes - fluxes_found

# scaler = 1e3
# positions_residuals = true_positions - positions_found
# r_residuals_rads = np.hypot(positions_residuals[:, :, 0], positions_residuals[:, :, 1])
# r_residuals = r2a(r_residuals_rads)


# '''View optimisation'''
# j = len(models_out)
# plt.figure(figsize=(16, 13))

# plt.subplot(3, 2, 1)
# plt.title("Log10 Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Log10 ADU")
# plt.plot(np.log10(np.array(losses)[:j]))

# plt.subplot(3, 2, 2)
# plt.title("Stellar Positions")
# plt.xlabel("Epochs")
# plt.ylabel("Positional Error (arcseconds)")
# plt.plot(r_residuals[:j])
# plt.axhline(0, c='k', alpha=0.5)

# plt.subplot(3, 2, 3)
# plt.title("Stellar Fluxes")
# plt.xlabel("Epochs")
# plt.ylabel("Flux Error (Photons)")
# plt.plot(flux_residuals[:j])
# plt.axhline(0, c='k', alpha=0.5)

# plt.subplot(3, 2, 4)
# plt.title("Zernike Coeff Residuals")
# plt.xlabel("Epochs")
# plt.ylabel("Residual Amplitude")
# plt.plot(coeff_residuals[:j])
# plt.axhline(0, c='k', alpha=0.5)

# plt.tight_layout()
# plt.show()


# '''Figure 5'''
# # OPDs
# true_opd = tel.ApplyBasisOPD.get_total_opd()
# opds_found = np.array([model.ApplyBasisOPD.get_total_opd() for model in models_out])
# found_opd = opds_found[-1]
# opd_residuls = true_opd - opds_found
# opd_rmse_nm = 1e9*np.mean(opd_residuls**2, axis=(-1,-2))**0.5

# vmin = np.min(np.array([true_opd, found_opd]))
# vmax = np.max(np.array([true_opd, found_opd]))

# # Coefficients
# true_coeff = tel.get(zernikes)
# found_coeff = models_out[-1].get(zernikes)
# index = np.arange(len(true_coeff))+4

# plt.figure(figsize=(20, 10))
# plt.suptitle("Optical Aberrations")

# plt.subplot(2, 2, 1)
# plt.title("RMS OPD residual")
# plt.xlabel("Epochs")
# plt.ylabel("RMS OPD (nm)")
# plt.plot(opd_rmse_nm)
# plt.axhline(0, c='k', alpha=0.5)

# plt.subplot(2, 2, 2)
# plt.title("Zernike Coefficient Amplitude")
# plt.xlabel("Index")
# plt.ylabel("Amplitude")
# plt.scatter(index, true_coeff, label="True Value")
# plt.scatter(index, found_coeff, label="Recovered Value", marker='x')
# plt.bar(index, true_coeff - found_coeff, label='Residual')
# plt.axhline(0, c='k', alpha=0.5)
# plt.legend()

# plt.subplot(2, 3, 4)
# plt.title("True OPD")
# plt.imshow(true_opd)
# plt.colorbar()

# plt.subplot(2, 3, 5)
# plt.title("Found OPD")
# plt.imshow(found_opd)
# plt.colorbar()

# plt.subplot(2, 3, 6)
# plt.title("OPD Residual")
# plt.imshow(true_opd - found_opd, vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.show()


# '''Figure 6'''
# # OPDs
# true_opd = tel.ApplyBasisOPD.get_total_opd()
# opds_found = np.array([model.ApplyBasisOPD.get_total_opd() for model in models_out])
# found_opd = opds_found[-1]
# opd_residuls = true_opd - opds_found
# opd_rmse_nm = 1e9*np.mean(opd_residuls**2, axis=(-1,-2))**0.5

# vmin = np.min(np.array([true_opd, found_opd]))
# vmax = np.max(np.array([true_opd, found_opd]))

# # Coefficients
# true_coeff = tel.get(zernikes)
# found_coeff = models_out[-1].get(zernikes)
# index = np.arange(len(true_coeff))+4

# plt.figure(figsize=(20, 10))
# plt.suptitle("Optical Aberrations")

# plt.subplot(2, 2, 1)
# plt.title("RMS OPD residual")
# plt.xlabel("Epochs")
# plt.ylabel("RMS OPD (nm)")
# plt.plot(opd_rmse_nm)
# plt.axhline(0, c='k', alpha=0.5)

# plt.subplot(2, 2, 2)
# plt.title("Zernike Coefficient Amplitude")
# plt.xlabel("Index")
# plt.ylabel("Amplitude")
# plt.scatter(index, true_coeff, label="True Value")
# plt.scatter(index, found_coeff, label="Recovered Value", marker='x')
# plt.bar(index, true_coeff - found_coeff, label='Residual')
# plt.axhline(0, c='k', alpha=0.5)
# plt.legend()

# plt.subplot(2, 3, 4)
# plt.title("True OPD")
# plt.imshow(true_opd)
# plt.colorbar()

# plt.subplot(2, 3, 5)
# plt.title("Found OPD")
# plt.imshow(found_opd)
# plt.colorbar()

# plt.subplot(2, 3, 6)
# plt.title("OPD Residual")
# plt.imshow(true_opd - found_opd, vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.show()



# '''Figure 7'''
# # calculate the mask where there was enough flux to infer the flat field
# thresh = 2500
# fmask = data.mean(0) >= thresh

# out_mask = np.where(data.mean(0) < thresh)
# in_mask = np.where(data.mean(0) >= thresh)

# data_tile = np.tile(data.mean(0), [len(models_out), 1, 1])
# in_mask_tiled = np.where(data_tile >= thresh)

# # calculate residuals
# pr_residuals = pix_response[in_mask] - flatfields_found[-1][in_mask]

# # for correlation plot
# true_pr_masked = pix_response.at[out_mask].set(1)
# found_pr_masked = flatfields_found[-1].at[out_mask].set(1)

# # FF Scatter Plot
# data_sum = data.sum(0) # [flux_mask]
# colours = data_sum.flatten()
# ind = np.argsort(colours)
# colours = colours[ind]

# pr_true_flat = true_pr_masked.flatten()
# pr_found_flat = found_pr_masked.flatten()

# pr_true_sort = pr_true_flat[ind]
# pr_found_sort = pr_found_flat[ind]

# # Errors
# pfound = flatfields_found[in_mask_tiled].reshape([len(models_out), len(in_mask[0])])
# ptrue = pix_response[in_mask]
# pr_res = ptrue - pfound
# masked_error = np.abs(pr_res).mean(-1)

# plt.figure(figsize=(20, 10))
# plt.subplot(2, 3, (1,2))
# plt.title("Pixel Response")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Sensitivity Error")
# plt.plot(masked_error)
# plt.axhline(0, c='k', alpha=0.5)

# # FF Scatter Plot
# data_sum = data.sum(0)
# colours = data_sum.flatten()
# ind = np.argsort(colours)
# colours = colours[ind]

# pr_true_flat = true_pr_masked.flatten()
# pr_found_flat = found_pr_masked.flatten()

# pr_true_sort = pr_true_flat[ind]
# pr_found_sort = pr_found_flat[ind]

# plt.subplot(2, 3, 3)
# plt.plot(np.linspace(0.8, 1.2), np.linspace(0.8, 1.2), c='k', alpha=0.75)
# plt.scatter(pr_true_sort, pr_found_sort, c=colours, alpha=0.5)
# plt.colorbar()
# plt.title("Sensitivity Residual")
# plt.ylabel("Recovered Sensitivity")
# plt.xlabel("True Sensitivity")

# plt.subplot(2, 3, 4)
# plt.title("True Pixel Response")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.imshow(true_pr_masked)
# plt.colorbar()

# vmin = np.min(pix_response)
# vmax = np.max(pix_response)

# plt.subplot(2, 3, 5)
# plt.title("Found Pixel Response")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.imshow(found_pr_masked, vmin=vmin, vmax=vmax)
# plt.colorbar()

# plt.subplot(2, 3, 6)
# plt.title("Pixel Response Residual")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.imshow(true_pr_masked - found_pr_masked, vmin=-0.2, vmax=0.2)
# plt.colorbar()

# plt.show()