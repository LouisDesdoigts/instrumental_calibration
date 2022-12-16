rule make_model_and_data:
    input:
        "src/data/mask.npy"
    output:
        directory("src/data/make_model_and_data/")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/make_model_and_data.py"

rule optimise:
    input:
        rules.make_model_and_data.output,
    output:
        directory("src/data/optimise")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/optimise.py"

rule calc_errors:
    input:
        rules.make_model_and_data.output,
        rules.optimise.output,
    output:
        directory("src/data/calc_errors")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/calc_errors.py"

rule divergence:
    input:
        "src/data/mask.npy"
    output:
        directory("src/data/divergence")
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/divergence.py"

rule plot_optics:
    input:
        rules.make_model_and_data.output,
    output:
        "src/tex/figures/optics.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_optics.py"

rule plot_FF:
    input:
        rules.optimise.output,
    output:
        "src/tex/figures/FF.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_FF.py"

rule plot_astro_params:
    input:
        rules.calc_errors.output,
        rules.optimise.output,
    output:
        "src/tex/figures/astro_params.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_astro_params.py"

rule plot_aberrations:
    input:
        rules.calc_errors.output,
        rules.optimise.output,
    output:
        "src/tex/figures/aberrations.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_aberrations.py"

rule plot_data_resid:
    input:
        rules.make_model_and_data.output,
        rules.optimise.output,
    output:
        "src/tex/figures/data_resid.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_data_resid.py"

rule plot_divergence:
    input:
        rules.divergence.output,
    output:
        "src/tex/figures/divergence.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_divergence.py"

rule compute_rms_in:
    input:
        rules.calc_errors.output,
        rules.optimise.output,
    output:
        "src/tex/output/rms_opd_in.txt"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_aberrations.py"

rule compute_rms_resid:
    input:
        rules.calc_errors.output,
        rules.optimise.output,
    output:
        "src/tex/output/rms_opd_resid.txt"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_aberrations.py"