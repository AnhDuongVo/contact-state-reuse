# Reproducibility

## Environment

* Python 3.10 to 3.13, `pip install -e ".[dev]"`. Tested with MuJoCo 3.x, PyTorch 2.x (CPU), Gymnasium 1.x, NumPy 2.x.
* Rendering is optional and only used by `ContactInsertionEnv.render`. Headless machines need `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, with `libosmesa6`). Without either, `render()` raises an error that says so; nothing else depends on rendering.

## Seeds

* Every random draw uses an explicit generator. Environment parameters, start poses and sensor noise come from `env.reset(seed=...)`; the expert, DART noise and evaluation-time corruption have their own derived seeds (`utils.seeding.derive_seed`).
* `seed_everything(seed)` seeds Python, NumPy and PyTorch and calls `torch.use_deterministic_algorithms(True, warn_only=True)`.
* On CPU, repeated runs with the same seed give identical datasets, identical training curves and identical evaluation results (tests check environment and rollout determinism). On GPU, some kernels are not deterministic; expect small differences in losses and possibly in episode outcomes.

## Data and outputs

* Datasets record their seed, configuration, environment and expert settings and the git commit (`meta.json`).
* Each run directory (`outputs/<experiment>_<scale>/<timestamp>/`) contains `config.yaml` (the merged configuration actually used), checkpoints with model and training configuration, normaliser and episode split, `episode_metrics.csv`, `results.csv`, `metrics.json` and figures.
* Experiment datasets are cached in `data/experiments/` under names that encode domain, severity, size, seed, perturbation fraction and noise; rerunning reuses them.

## Scales

| config | seeds | train episodes | updates | eval episodes per cell |
|---|---|---|---|---|
| `demo` | 1 | 60 | 200 | 10 |
| experiments, `--scale quick` | 1 (D: 2) | 120 | 300 | 20 to 30 |
| experiments, `--scale full` | 3 | 400 | 3000 | 50 to 100 |

Quick runs check the pipeline and give a first look; they are too small for conclusions. Report full-scale results with seed-level variation.

## Checklist before reporting a number

1. It comes from a `results.csv` or JSON written by the pipeline, not from a log line.
2. The interval and `n` are reported with it.
3. The comparison is paired (same evaluation seed) and uses the same training budget.
4. Seed-level variation is reported for full-scale runs.
5. The configuration file in the run directory matches what is described.
