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
        'src/data/optimise/true_prf_sorted.npy'
        'src/data/optimise/found_prf_sorted.npy'
        'src/data/optimise/colours.npy'
        "src/data/optimise/pixel_response_resid_counts.npy"
        "src/data/optimise/pixel_response_resid_bins.npy"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_FF.py"

rule astro_params:
    input:
        'src/data/make_model_and_data/instrument.p'
        'src/data/optimise/models_out.p'
        'src/data/calc_errors/cov_mat.npy'
    output:
        directory("src/data/astro_params")
    conda:
        "environment.yml"
    script:
        "src/scripts/astro_params.py"

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
