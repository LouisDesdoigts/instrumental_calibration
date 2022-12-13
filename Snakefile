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

rule compute_answer:
    input:
        'src/data/make_model_and_data/instrument.p'
        'src/data/optimise/models_out.p'
        'src/data/calc_errors/cov_mat.npy'
    output:
        "src/tex/output/rms_opd_resid.txt"
        "src/tex/output/rms_opd_in.txt"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_aberrations.py"
