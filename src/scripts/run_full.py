import sys
import os
import time as t
import paths
from lib import mkdir, format_time, run_script
# from lib import make_model_and_data, optimise, process, calc_errors
from lib import optimise, calc_errors

fluxes = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
PRFdevs = [0.0001, 0.001, 0.01, 0.1]

mkdir(paths.data)

for i in range(len(fluxes)):
    flux = fluxes[i]
    for j in range(len(PRFdevs)):
        PRFdev = PRFdevs[j]
        sub_dir = f"flux_{fluxes[i]:.0e}_PRFdev_{PRFdevs[j]:.0e}"

        print(f"Flux, PRFdev: {flux:.0e}, {PRFdev:.1}")
        optimise(flux, PRFdev)
        if flux == 1e8 and PRFdev == 0.1:
            calc_errors(flux, PRFdev)

# # Plotting
# run_script('plot_astro_params', sub_dir)
# run_script('plot_aberrations',  sub_dir)
# run_script('plot_data_resid',   sub_dir)
# run_script('plot_optics',       sub_dir)
# run_script('plot_FF',           sub_dir)
# run_script('plot_noise',        sub_dir)