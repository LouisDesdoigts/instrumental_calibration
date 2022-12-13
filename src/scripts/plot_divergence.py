import jax.numpy as np
from dLux.utils import radians_to_arcseconds as r2a
import paths
import dill as p
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120

positions = 'PointSource.position'
fluxes = 'PointSource.flux'
zernikes = 'ApplyBasisOPD.coefficients'
flatfield = 'ApplyPixelResponse.pixel_response'

fluxes_in = np.load(paths.data / "divergence/divergence_fluxes_in.npy")
models_out = p.load(open(paths.data / 'divergence/divergence_models_out.p', 'rb'))

# Positions
positions_found = np.array([model.get(positions) for model in models_out])
x_res, y_res = (np.zeros(2)- positions_found[-1]).T
r_res = np.hypot(x_res, y_res)

# Fluxes
final_fluxes = models_out[-1].get(fluxes)
res = np.abs(fluxes_in - final_fluxes[:, 0])
l1 = res/fluxes_in
xs = fluxes_in
diffs = (models_out[-1].get(fluxes) - models_out[-2].get(fluxes))/fluxes_in
sigma = 1

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Positional Residual")
plt.scatter(fluxes_in, r2a(r_res))
plt.semilogx()
plt.axhline(0, c='k')
plt.xlabel("Flux")
plt.ylabel("Positional Residual (arcseconds)")

plt.subplot(1, 2, 2)
plt.title("Fractional Flux Error $|(\hat{f} - f)/\hat{f}|$")
plt.errorbar(xs, l1, yerr=(sigma*(xs**0.5)/xs), fmt='o')
plt.axhline(0, c='k')
plt.semilogx()
plt.xlabel("Flux")
plt.ylabel("Fractional Error")

plt.tight_layout()
plt.savefig(paths.figures / "divergence.pdf", dpi=300)