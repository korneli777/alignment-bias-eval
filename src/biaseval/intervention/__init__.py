"""Linear concept erasure: INLP (Ravfogel 2020), LEACE (Belrose 2023), and the
forward-hook wrapper that applies either method's projection at a chosen layer.
"""

from biaseval.intervention.hooks import ProjectionHook
from biaseval.intervention.inlp import (
    LeaceResult,
    NullspaceResult,
    fit_inlp,
    fit_leace,
)
from biaseval.intervention.sanity import (
    lm_perplexity,
    verify_nullification,
)

__all__ = [
    "LeaceResult",
    "NullspaceResult",
    "ProjectionHook",
    "fit_inlp",
    "fit_leace",
    "lm_perplexity",
    "verify_nullification",
]
