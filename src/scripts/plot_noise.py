import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import jax.scipy as jsp
import matplotlib.pyplot as plt
from zodiax.experimental.serialisation import serialise, deserialise
from observation import MultiImage
import dLux as dl
from lib import weighted_pearson_corr_coeff

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

fluxes = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
PRFdevs = [0.0001, 0.001, 0.01, 0.1]

flux_err, pos_err, zern_err, corrs = [], [], [], []
for i in range(len(fluxes)):
    flux = fluxes[i]
    flux_err_, pos_err_, zern_err_, corrs_ = [], [], [], []
    for j in range(len(PRFdevs)):
        PRFdev = PRFdevs[j]
        sub_dir = f"flux_{fluxes[i]:.0e}_PRFdev_{PRFdevs[j]:.0e}"

        data = np.load(paths.data / f"make_model_and_data/{sub_dir}/data.npy")
        tel = deserialise(paths.data / f'make_model_and_data/{sub_dir}/instrument.zdx')
        model = deserialise(paths.data / f'optimise/{sub_dir}/final_model.zdx')


        # PRF Correlations
        flatfield = 'PRF.pixel_response'
        corr = weighted_pearson_corr_coeff(
            model.get(flatfield), tel.get(flatfield), data.sum(0))
        corrs_.append(corr)


        # Fluxes
        true_fluxes = tel.Source.flux
        found_fluxes = model.Source.flux
        expected_err = np.mean(np.sqrt(true_fluxes))
        MAE = np.mean(np.abs(true_fluxes - found_fluxes)/np.sqrt(true_fluxes))
        flux_err_.append(MAE)


        # Positions
        true_pos = tel.Source.position
        found_pos = model.Source.position
        mean_wl = tel.Source.spectrum.wavelengths.mean()
        lamd = mean_wl/tel.CreateWavefront.diameter
        radial_error = 1/np.pi * np.sqrt(2/true_fluxes) * lamd # Radians
        PE = np.abs(true_pos - found_pos)
        APE = np.hypot(PE[:, 0], PE[:, 1])
        rel_PE = np.mean(APE/radial_error)
        pos_err_.append(rel_PE)


        # Aberrations
        basis = tel.Aberrations.basis
        true_coefficients = tel.Aberrations.coefficients
        found_coefficients = model.Aberrations.coefficients
        true_per_mode_aberration = basis * true_coefficients[:, None, None]
        found_per_mode_aberration = basis * found_coefficients[:, None, None]
        per_mode_opd_residual = true_per_mode_aberration - found_per_mode_aberration
        per_mode_opd_residual_radians = dl.utils.opd_to_phase(per_mode_opd_residual, mean_wl)
        RMS_per_mode_rad = np.sqrt(np.mean(per_mode_opd_residual_radians**2, axis=(1, 2)))
        expected_zern_err = 1/(np.sqrt(true_fluxes).mean())
        rel_zern_err = (RMS_per_mode_rad.mean()/expected_zern_err)
        zern_err_.append(rel_zern_err)


    corrs.append(corrs_)
    flux_err.append(flux_err_)
    pos_err.append(pos_err_)
    zern_err.append(zern_err_)

corrs = np.array(corrs)
flux_err = np.array(flux_err)
pos_err = np.array(pos_err)
zern_err = np.array(zern_err)

PRF_percentages = np.array(PRFdevs) * 100
Fluxes = np.round(np.log10(np.array(fluxes)),  decimals=0).astype(int)


plt.figure(figsize=(10, 12))
plt.subplot(2, 2, 1)
plt.title("PRF Weighted Correlation")
plt.imshow(corrs, vmin=0, vmax=1)
plt.yticks(range(len(Fluxes)), Fluxes)
plt.xticks(range(len(PRF_percentages)), PRF_percentages)
plt.ylabel("Log10 Flux (Photons)")
plt.xlabel("PRF Deviation (%)")
cbar = plt.colorbar()
cbar.set_label("Correlation Coefficient")

plt.subplot(2, 2, 2)
plt.imshow(np.log10(zern_err))
plt.title("Relative Zernike Error")
plt.yticks(range(len(fluxes)), Fluxes)
plt.xticks(range(len(PRF_percentages)), PRF_percentages)
plt.ylabel("Log10 Flux (Photons)")
plt.xlabel("PRF Deviation (%)")
cbar = plt.colorbar()
cbar.set_label("Log10 Relative Error")

plt.subplot(2, 2, 3)
plt.title("Relative Flux Error")
plt.imshow(np.log10(flux_err))
plt.yticks(range(len(fluxes)), Fluxes)
plt.xticks(range(len(PRF_percentages)), PRF_percentages)
plt.ylabel("Log10 Flux (Photons)")
plt.xlabel("PRF Deviation (%)")
cbar = plt.colorbar()
cbar.set_label("Log10 Relative Error")

plt.subplot(2, 2, 4)
plt.title("Relative Position Error")
plt.imshow(np.log10(pos_err))
plt.yticks(range(len(fluxes)), Fluxes)
plt.xticks(range(len(PRF_percentages)), PRF_percentages)
plt.ylabel("Log10 Flux (Photons)")
plt.xlabel("PRF Deviation (%)")
cbar = plt.colorbar()
cbar.set_label("Log10 Relative Error")

plt.tight_layout()
plt.savefig(paths.figures / "noise_performance.pdf", dpi=300)