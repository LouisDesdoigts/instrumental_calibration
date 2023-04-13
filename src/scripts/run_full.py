import sys
import os
import time as t
import paths
from lib import mkdir, format_time, run_script
from lib import make_model_and_data, optimise, process, calc_errors

fluxes = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
PRFdevs = [0.0001, 0.001, 0.01, 0.1]

# fluxes = [1e8]
# PRFdevs = [0.1]

for i in range(len(fluxes)):
    flux = fluxes[i]
    for j in range(len(PRFdevs)):
        PRFdev = PRFdevs[j]
        sub_dir = f"flux_{fluxes[i]:.0e}_PRFdev_{PRFdevs[j]:.0e}"

        # print()
        print(f"Flux, PRFdev: {flux:.0e}, {PRFdev:.1}")

        # make_model_and_data(flux, PRFdev)
        # optimise(flux, PRFdev)
        # process(flux, PRFdev)
        # calc_errors(flux, PRFdev)

        # # Calculation
        # run_script('make_model_and_data', sub_dir, True)
        # # run_script('optimise',            sub_dir, True)
        # run_script('optimise2',           sub_dir, True)
        # run_script('process',             sub_dir, True)
        # run_script('calc_errors',         sub_dir, True)

        # # Plotting
        # run_script('plot_progress',     sub_dir)
        # run_script('plot_astro_params', sub_dir)
        # run_script('plot_aberrations',  sub_dir)
        # run_script('plot_data_resid',   sub_dir)
        # run_script('plot_optics',       sub_dir)
        # run_script('plot_FF',           sub_dir)
        
        # mkdir(paths.figures / f"summary/Flux_{fluxes[i]:.0e}")
        # run_script('plot_summary', sub_dir)