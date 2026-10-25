"""Representation analysis: latents, probes, similarity, reuse, ablation, plots."""

from .ablation import clamp_fn, dimension_sets, rank_contact_dims, rank_movement_dims
from .latents import LatentSet, align, extract_latents
from .probes import ProbeResult, cross_domain_probe, grouped_cv_probe, probe_weights
from .reuse import ReuseResult, reuse_score
from .similarity import cka_blocks, cka_null, linear_cka, phase_matched_blocks
from .stats import spearman_with_ci

__all__ = [
    "LatentSet",
    "ProbeResult",
    "ReuseResult",
    "align",
    "cka_blocks",
    "cka_null",
    "clamp_fn",
    "cross_domain_probe",
    "dimension_sets",
    "extract_latents",
    "grouped_cv_probe",
    "linear_cka",
    "phase_matched_blocks",
    "probe_weights",
    "rank_contact_dims",
    "rank_movement_dims",
    "reuse_score",
    "spearman_with_ci",
]
