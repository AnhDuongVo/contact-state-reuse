"""Linear decoding of simulator contact modes from frozen latents.

Cross-validation folds are built with ``GroupKFold`` over episode ids, so all
time steps of an episode are either in the training fold or in the test fold.
This prevents temporal leakage: neighbouring time steps are nearly identical
and would otherwise inflate accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..envs.labels import N_MODES


@dataclass
class ProbeResult:
    balanced_accuracy: float
    fold_scores: list[float]
    confusion: np.ndarray  # row-normalised [N_MODES, N_MODES]
    chance: float


def make_probe(C: float = 1.0, seed: int = 0):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    )


def _confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y, p, labels=np.arange(N_MODES)).astype(float)
    return cm / np.maximum(cm.sum(1, keepdims=True), 1.0)


def grouped_cv_probe(
    z: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, C: float = 1.0, seed: int = 0
) -> ProbeResult:
    """Episode-grouped k-fold decoding of ``y`` from ``z``."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("need at least two episodes for grouped cross-validation")
    k = min(n_splits, n_groups)
    preds = np.full(len(y), -1)
    scores = []
    for tr, te in GroupKFold(n_splits=k).split(z, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_probe(C, seed).fit(z[tr], y[tr])
        preds[te] = clf.predict(z[te])
        scores.append(float(balanced_accuracy_score(y[te], preds[te])))
    ok = preds >= 0
    return ProbeResult(
        float(balanced_accuracy_score(y[ok], preds[ok])), scores, _confusion(y[ok], preds[ok]), 1.0 / len(np.unique(y))
    )


def cross_domain_probe(
    z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, C: float = 1.0, seed: int = 0
) -> ProbeResult:
    """Train on one domain, test on another (no shared episodes by construction)."""
    clf = make_probe(C, seed).fit(z_train, y_train)
    p = clf.predict(z_test)
    return ProbeResult(
        float(balanced_accuracy_score(y_test, p)), [], _confusion(y_test, p), 1.0 / len(np.unique(y_test))
    )


def probe_weights(z: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 0) -> np.ndarray:
    """Per-dimension importance: sum over classes of |coefficient| on standardised latents."""
    clf = make_probe(C, seed).fit(z, y)
    return np.abs(clf[-1].coef_).sum(0)


@dataclass
class PairedProbeResult:
    """Held-out-pair decoding: fit on source members of training pairs, test on
    source and shifted members of *held-out* pairs."""

    source_bacc: float
    shifted_bacc: float
    source_recall: dict[int, float]
    shifted_recall: dict[int, float]
    support_source: dict[int, int]
    support_shifted: dict[int, int]


def paired_heldout_probe(
    z_src: np.ndarray,
    y_src: np.ndarray,
    ep_src: np.ndarray,
    z_tgt: np.ndarray,
    y_tgt: np.ndarray,
    ep_tgt: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    seed: int = 0,
) -> PairedProbeResult:
    """Episode pairs (same episode id in both domains) are split into folds; the
    probe never sees either member of a test pair during fitting. Balanced
    accuracy is computed over the pooled out-of-fold predictions, on the same
    set of classes for both domains (classes present in both test sets)."""
    pairs = np.intersect1d(np.unique(ep_src), np.unique(ep_tgt))
    k = min(n_splits, len(pairs))
    folds = np.array_split(np.random.default_rng(seed).permutation(pairs), k)
    ps, pt = np.full(len(y_src), -1), np.full(len(y_tgt), -1)
    for f in folds:
        tr = ~np.isin(ep_src, f) & np.isin(ep_src, pairs)
        if len(np.unique(y_src[tr])) < 2:
            continue
        clf = make_probe(C, seed).fit(z_src[tr], y_src[tr])
        ms, mt = np.isin(ep_src, f), np.isin(ep_tgt, f)
        ps[ms] = clf.predict(z_src[ms])
        pt[mt] = clf.predict(z_tgt[mt])
    oks, okt = ps >= 0, pt >= 0
    classes = np.intersect1d(np.unique(y_src[oks]), np.unique(y_tgt[okt]))
    ks, kt = oks & np.isin(y_src, classes), okt & np.isin(y_tgt, classes)

    def rec(y, p):
        r = recall_score(y, p, labels=classes, average=None, zero_division=0)
        return {int(c): float(v) for c, v in zip(classes, r)}

    rs, rt = rec(y_src[ks], ps[ks]), rec(y_tgt[kt], pt[kt])
    return PairedProbeResult(
        float(np.mean(list(rs.values()))),
        float(np.mean(list(rt.values()))),
        rs,
        rt,
        {int(c): int((y_src[ks] == c).sum()) for c in classes},
        {int(c): int((y_tgt[kt] == c).sum()) for c in classes},
    )
