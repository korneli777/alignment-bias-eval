"""biaseval: stereotypical-bias evaluation for causal language models.

Submodules:
    benchmarks   CrowS-Pairs, StereoSet, BBQ, and IAT scoring.
    probing      Probe-dataset construction, activation extraction, per-layer
                 logistic-regression probes.
    intervention INLP, LEACE, forward-hook wrapper, sanity checks.
    analysis     Bootstrap CIs, paired effect sizes, regression with HC3 SEs.
"""

__version__ = "1.0.0"
