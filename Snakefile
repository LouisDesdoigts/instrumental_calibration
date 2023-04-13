rule run_full:
    output:
        # directory("src/data/make_model_and_data"),
        # directory("src/data/optimise"),
        # directory("src/data/process"),
        # directory("src/data/calc_errors")
        directory("src/data"),
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/run_full.py"

rule plot_optics:
    input:
        # rules.make_model_and_data.output,
        rules.run_full.output,
    output:
        "src/tex/figures/optics.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_optics.py"

rule plot_FF:
    input:
        # rules.optimise.output,
        rules.run_full.output,
    output:
        "src/tex/figures/FF.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_FF.py"

rule plot_astro_params:
    input:
        # rules.calc_errors.output,
        # rules.optimise.output,
        rules.run_full.output,
    output:
        "src/tex/figures/astro_params.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_astro_params.py"

rule plot_aberrations:
    input:
        # rules.calc_errors.output,
        # rules.optimise.output,
        rules.run_full.output,
    output:
        "src/tex/figures/aberrations.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_aberrations.py"

rule plot_data_resid:
    input:
        # rules.make_model_and_data.output,
        # rules.optimise.output,
        rules.run_full.output,
    output:
        "src/tex/figures/data_resid.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_data_resid.py"

rule plot_noise:
    input:
        rules.run_full.output,
    output:
        "src/tex/figures/noise.pdf"
    conda:
        "environment.yml"
    script:
        "src/scripts/plot_noise.py"