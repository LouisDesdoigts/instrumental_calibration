rule compute_answer:
    output:
        "src/tex/output/opd_in.txt"
    conda:
        "environment.yml"
    script:
        "src/scripts/gen_aberrations.py"

rule make_model_and_data:
    output:
        directory("src/data/make_model_and_data")
        # "src/data/pixel_response_counts.npy"
        # "src/data/pixel_response_bins.npy"
        # "src/data/plain_psf.npy"
        # "src/data/aberrated_psf.npy"
        # "src/data/data.npy"
        # "src/data/initial_psfs.npy"
        # "src/data/instrument.p"
        # "src/data/source.p"
        # "src/data/model.p"
    cache:
        True
    script:
        "src/scripts/make_model_and_data.py"

rule optimise:
    output:
        directory("src/data/optimise")
        # "src/data/losses.npy"
        # "src/data/models_out.p"
        # "src/data/true_prf_sorted.npy"
        # "src/data/found_prf_sorted.npy"
        # "src/data/colours.npy"
        # "src/data/pixel_response_resid_counts.npy"
        # "src/data/pixel_response_resid_bins.npy"
        # "src/data/final_psfs.npy"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/optimise.py"

rule calc_errors:
    output:
        directory("src/data/calc_errors")
        # "src/data/cov_mat.npy"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/calc_errors.py"

rule divergence:
    output:
        directory("src/data/divergence")
        # "src/data/divergence_fluxes_in.npy"
        # "src/data/divergence_models_out.p"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/divergence.py"


