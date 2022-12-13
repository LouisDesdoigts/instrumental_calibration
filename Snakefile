rule make_model_and_data:
    output:
        directory("src/data/make_model_and_data")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/make_model_and_data.py"

rule optimise:
    output:
        directory("src/data/optimise")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/optimise.py"

rule calc_errors:
    output:
        directory("src/data/calc_errors")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/calc_errors.py"

rule divergence:
    output:
        directory("src/data/divergence")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/divergence.py"

rule plot_FF:
    input:
        "src/data/optimise/true_prf_sorted.npy"
        "src/data/optimise/found_prf_sorted.npy"
        "src/data/optimise/colours.npy"
        "src/data/optimise/pixel_response_resid_counts.npy"
        "src/data/optimise/pixel_response_resid_bins.npy"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_FF.py"

rule plot_astro_params:
    input:
        "src/data/make_model_and_data/instrument.p"
        "src/data/optimise/models_out.p"
        "src/data/calc_errors/cov_mat.npy"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_astro_params.py"

rule plot_aberrations:
    input:
        "src/data/make_model_and_data/instrument.p"
        "src/data/optimise/models_out.p"
        "src/data/calc_errors/cov_mat.npy"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_aberrations.py"

rule plot_data_resid:
    input:
        "src/data/make_model_and_data/data.npy"
        "src/data/optimise/final_psfs.npy"
        "src/data/make_model_and_data/initial_psfs.npy"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_data_resid.py"

rule plot_optics:
    input:
        "src/data/make_model_and_data/instrument.p"
        "src/data/make_model_and_data/wavelengths.npy"
        "src/data/make_model_and_data/plain_psf.npy"
        "src/data/make_model_and_data/aberrated_psf.npy"
        "src/data/make_model_and_data/pixel_response_counts.npy"
        "src/data/make_model_and_data/pixel_response_bins.npy"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_optics.py"

rule plot_divergence:
    input:
        "src/data/divergence/divergence_fluxes_in.npy"
        "src/data/divergence/divergence_models_out.p"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_divergence.py"

# rule compute_answer:
#     input:
#         'src/data/make_model_and_data/instrument.p'
#         'src/data/optimise/models_out.p'
#         'src/data/calc_errors/cov_mat.npy'
#     output:
#         "src/tex/output/rms_opd_resid.txt"
#         "src/tex/output/rms_opd_in.txt"
#     conda:
#         "environment.yml"
#     script:
#         "src/scripts/plot_aberrations.py"
