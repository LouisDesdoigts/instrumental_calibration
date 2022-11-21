rule compute_answer:
    output:
        "src/tex/output/opd_in.txt"
    script:
        "src/scripts/gen_aberrations.py"