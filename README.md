# When Does Force Feedback Help? Predictive Objectives and Contact-Onset Force Use

*Repository: `contact-state-reuse`, release v0.2.0 (revised after review).*

**Question.** In a simulated planar insertion task, when does force help a learned policy: as an input, as a target that a predictive auxiliary loss learns to predict, or both, and at which moments of the insertion? Every condition uses the same recurrent network and the same demonstrations, so only the role of force and the training losses change. All results are from simulation and say nothing yet about real robots.

**Short answer (v0.2, 5 seeds per condition).**

* Predicting the next observation and the next 0.25 s of force made training more reliable and lowered contact forces: success 0.96 vs 0.78 and peak force 5.5 vs 11.4 N relative to behaviour cloning with the same inputs; +0.07 success and -3.6 N relative to a behaviour-cloning baseline with a tuned learning rate.
* Force was most useful as a training target. A model that predicts force but never receives it reached 0.96 success (0.98 under sensor faults). Live force input then lowered peak force by 1.4 to 2.4 N on a healthy sensor and raised it by 3.8 N on a faulty one, unless the model was trained with simulated faults.
* The lower forces came mostly from the compliance the policy commands; with compliance fixed by the harness, predictive and behaviour-cloning policies reached similar forces.
* The trained policies were most sensitive to losing the force reading in the first 0.3 s of contact: +1.8 N, against +0.4 N for an equally long window 0.3 s later that removed a similar amount of force signal.
* The pattern held for a second, non-privileged demonstrator and on a second task (hole search), where force as an input became essential.
* Exploratory: under a shift that requires new behaviour (hole search), learned representations were not more stable than an untrained network's, and stability did not predict adaptation. The v0.1 "fine-tuning dip" is a small-data effect of behaviour cloning that the predictive models avoided.

Full write-up: [docs/whitepaper/whitepaper.pdf](docs/whitepaper/whitepaper.pdf) ([Markdown](docs/whitepaper/whitepaper.md)). Point-by-point response to the reviews of v0.1: [docs/whitepaper/response_to_reviewers.pdf](docs/whitepaper/response_to_reviewers.pdf). The first version is kept as [whitepaper_v1](docs/whitepaper/whitepaper_v1.pdf) with its results in [`results/medium/`](results/medium/).

![pipeline](assets/architecture.svg)

## What changed in v0.2

The first version compared models of equal parameter count but different trunk widths, let prediction losses see failed episodes that behaviour cloning did not, used 3 seeds with intervals over evaluation episodes only, and was framed around "contact-state reuse". v0.2 re-runs the study:

* one fixed network for every condition; losses switched on or off; all losses trained on the same successful demonstrations;
* a 2 x 2 x 2 factorial (force input, observation target, force target) plus 25 control conditions (failure data, reconstruction, stop-gradient, separate predictive core, loss weights, learning rates, capacity, previous action, force transforms, DART noise, fault-aware training);
* 5 seeds; intervals from a crossed bootstrap over training seeds and evaluation episodes;
* timing interventions at onset and at later windows, hold-vs-zero, onset-only; fixed-compliance runs;
* a second demonstrator, a second task, and a scripted non-privileged controller as a baseline;
* a redesigned, exploratory reuse and adaptation analysis with reference representations.

Two v0.1 claims did not replicate and are withdrawn: that force as an input harms behaviour cloning, and that the onset window "accounts for 85 to 94 %" of the force effect.

## Questions and outcomes

| | Statement | Outcome (v0.2) |
|---|---|---|
| H1 | Force improves success and safety, mostly under physical shifts | **Not supported as stated.** The effect of force depended on its role and the objective; force inputs were harmful under sensor faults |
| H2 | Predictive training gives latents that encode contact phase better | **Not supported.** Within-source decoding 0.66 (full model) to 0.70 (behaviour cloning) |
| H3 | Contact-state representations are reused across a shift | **Shift-dependent.** World-model latents above raw observations in contact phases for the tight fit; below an untrained network for the search shift |
| H4 | More reuse goes with faster, safer adaptation | **Not supported (exploratory).** rho -0.34 [-0.74, +0.20] across 20 trained models |
| H5 | Force matters specifically around first contact | **Supported (onset sensitivity).** Onset window +1.8 N vs +0.4 N for a later window removing similar signal |
| Q1 | Does prediction help on identical data; force as input or target? | Prediction helps (+0.18 vs BC with force, 5/5 seeds); force mainly as a target (+0.22 without force input) |
| Q2 | Failure data, generic auxiliary losses, weak baseline? | Not failure data; prediction beats reconstruction; smaller but present vs a tuned baseline |
| Q3 | How are forces lowered? | Mostly through the commanded compliance |
| Q4 | Other demonstrator or task? | Pattern holds; force input essential only for hole search |

## Environment

`ContactInsertionEnv` (Gymnasium API, MuJoCo physics): a rectangular peg moves in the x-z plane (tip x, tip z, rotation about y) towards a chamfered slot. Each joint is a PD actuator around a set-point; the action moves the set-point and sets a compliance level that scales the PD stiffness. Episodes pass through approach, first contact, alignment, insertion and either success or jam/failure.

| Modality | Dimensions | Content | Policy input |
|---|---|---|---|
| exteroception | 4 | estimated slot entrance relative to the peg tip (x, z), estimated relative angle, estimate-valid flag; a low-dimensional stand-in for a camera pose estimate, with calibration offset, noise and occlusion | yes |
| proprioception | 7 | joint positions, joint velocities, current compliance level (joint efforts optional, off by default) | yes |
| force | 3 | planar contact wrench (Fx, Fz, Ty) summed from MuJoCo contact forces, after a sensor model with noise, bias, delay and dropout | yes (masked in no-force conditions) |
| analysis labels | 6 classes + scalars | `contact_mode`, `first_contact`, `success`, `jam`, `insertion_depth`, `contact_force_norm` | **never** |

The force signal is an idealised external wrench at the contacts, not a wrist force/torque sensor model and not motor-current sensing. Peak contact force is the maximum over 2 ms physics steps of the summed planar contact force; a summed-contact-magnitude metric (`peak_force_sum`, no cancellation between opposing contacts) is reported alongside.

Domains (`src/contact_state_reuse/envs/domains.py`): `source`, `shifted_friction`, `tight_clearance`, `sensor_noise`, `occlusion` (the five evaluated domains), `search` (5 to 9 mm lateral error of the pose estimate; second task), composites such as `search+tight_clearance`, and `mixed`. Shifts can be applied at a graded `severity` in [0, 1].

Demonstrators (`envs/expert.py`): the privileged scripted expert (true slot pose and contact wrench; main data), a non-privileged variant that perceives only the observation vector (`privileged=False`; second demonstrator and baseline controller), and a search variant for the second task (`search=True`).

## Models

Every v0.2 condition is the same network (`PredictiveContactWorldModel`, GRU 66 units, 61,469 parameters): fusion encoder with per-modality embeddings and masks, GRU over observation and previous action, action head, and prediction heads for the next observation and the next 5 force readings. Conditions switch losses on or off (`lambda_obs`, `lambda_force`), mask the force input, or apply one of the controls: `lambda_recon` (reconstruction of the current observation), `aux_stopgrad`, `aux_core` (prediction heads on a separate GRU), `use_prev_action`, `force_transform` (`log`, `binary`), and the training options `episode_filter` and `force_aug` (fault-aware training). With both prediction weights at 0 the network is a sequence behaviour-cloning policy.

"World model" in the class name means a predictive auxiliary model in a restricted sense: it predicts future observations and forces from history and the policy reads the same latent, but it is not used for planning or imagined rollouts. `ReactivePolicy` and `SequencePolicy` (v0.1 baselines) remain available.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis]"          # or: pip install -r requirements-lock.txt && pip install -e . --no-deps

make test                                  # about 25 s
make demo                                  # small end-to-end run with figures, about 1 minute

# one experiment at small scale
python scripts/run_experiment.py configs/experiments/E_factorial.yaml --scale quick

# all v0.2 experiments, summaries and figures (about 6 hours on 2 CPU cores)
bash scripts/run_revision.sh
python scripts/analyze_force_input.py --run outputs/E_factorial_medium/run1 --out results/revision
python scripts/summarize_revision.py --out results/revision
python scripts/make_revision_figures.py --results results/revision --out docs/whitepaper/figures
python docs/whitepaper/build_pdf.py
```

Long experiments can be split over processes with `--run-dir <dir> --shard i/n` and aggregated with a final call without `--shard`; finished models and evaluations are cached, so interrupted runs resume.

## Runtime and resources

Measured on a 2-core cloud CPU without GPU.

| Command | Time |
|---|---|
| `make test` | about 25 s |
| one model (1,500 updates) plus evaluation on five domains | about 100 s on one core |
| experiment E (33 conditions x 5 seeds) | about 2.5 hours on 2 cores |
| experiments T and L (evaluation only) | about 20 and 10 minutes |
| experiments G1 and G2 (8 conditions x 5 seeds each) | about 40 minutes each |
| experiment R (20 pre-trained models, 380 fine-tuning runs) | about 2 hours |
| v0.1 experiments A to D, medium | about 75 minutes |

## Results

All v0.2 numbers come from `scripts/summarize_revision.py` and are committed under [`results/revision/`](results/revision/) (per-episode CSVs in `runs/`, tables in [`SUMMARY.md`](results/revision/SUMMARY.md), key numbers in `key_numbers.json`). Intervals are 95 % crossed seed x episode bootstrap intervals (5 training seeds x 50 evaluation episodes per domain).

**Experiment E, pooled over the five domains (1,250 episodes per row):**

| Condition | Force input | Obs. target | Force target | Success | Peak force [N] |
|---|---|---|---|---|---|
| BC | | | | 0.64 [0.46, 0.81] | 11.2 [10.0, 12.5] |
| BC + force | x | | | 0.78 [0.70, 0.85] | 11.4 [9.3, 14.1] |
| BC + force, lr 3e-4 | x | | | 0.89 [0.82, 0.95] | 9.1 [8.3, 9.9] |
| force target | | | x | 0.87 [0.82, 0.91] | 9.5 [7.9, 11.0] |
| obs prediction + force | x | x | | 0.91 [0.84, 0.97] | 8.0 [5.1, 10.9] |
| obs + force targets, no force input | | x | x | 0.96 [0.94, 0.98] | 6.2 [5.3, 7.0] |
| obs + force targets + force input | x | x | x | 0.96 [0.93, 0.98] | 5.5 [4.8, 6.1] |
| same, fault-aware training | x | x | x | 0.98 [0.96, 0.99] | 5.6 [4.2, 7.3] |

**Timing (experiment T, predictive model, source domain):** removing the force reading for 0.3 s at contact onset +1.8 N [1.1, 2.6]; 0.3 s later +0.4 N [0.2, 0.8] (similar amount of signal removed); 0.6 s later +0.1 N; random window 0.0 N; whole episode +4.0 N [2.6, 5.5].

**Generalisation (G1, G2):** with the non-privileged demonstrator the force-target model without force input was the best learned policy (0.98, 4.7 N); on hole search force input was essential (BC 0.14 without, 0.67 with) and the full predictive model reached 0.73 (0.90 in the plain search domain).

**Exploratory (R):** under the search shift (zero-shot success 0.11 to 0.22), 40 fine-tuning demonstrations raised every family to 0.60 to 0.68, with no family adapting faster; reuse did not predict the gain (rho -0.34 [-0.74, +0.20], n = 20). Fine-tuning behaviour cloning on 5 source demonstrations lowered its source success from 0.88 to 0.48, while the predictive models stayed at 0.95 to 0.97.

These are simulation results with simple planar tasks. They describe what these models learned from these demonstrations and do not establish real-robot behaviour.

## Scientific rigour

* **One network.** Every v0.2 condition uses the identical architecture and initialisation per seed; only losses, input masks and training data options change. A test checks that `step` and `forward` agree for every variant.
* **Identical data.** All losses train on the same successful demonstrations; failure exposure is a separate, explicit factor.
* **Training-run uncertainty.** 5 seeds per condition; crossed seed x episode bootstrap with paired cells (`evaluation/metrics.py`); per-seed means reported.
* **Episode-level splits.** Train/validation splits and probe cross-validation are grouped by episode, never by time step (tested).
* **Paired evaluation.** Evaluation seeds are disjoint from data seeds, and episode *i* of an evaluation set is identical across conditions.
* **Interventions with accounting.** Force interventions record whether the window was reached and how much force signal was removed.
* **References and nulls for similarity.** Reuse scores are compared with permutation nulls (full distribution) and with raw-observation, proprioception, task-progress and untrained-network references.
* **Labels are for analysis.** Contact modes are computed from privileged simulator state and never given to a v0.2 policy.

## Limitations and next steps

* Simulation only; the force signal is an ideal contact wrench, not actuator-current or wrist-sensor force.
* Small planar tasks, low-dimensional "vision", scripted demonstrators, 300 demonstrations and 61k-parameter models.
* Five seeds; some behaviour-cloning comparisons remain inconclusive.
* The scripted non-privileged controller is a strong baseline on these tasks; this repository compares learned policies with each other.

Next steps: actuator-derived force, data-scale studies, fault-aware sensing with health estimation, hardware onset-window tests, image observations. More in [docs/limitations.md](docs/limitations.md).

## Repository layout

```
configs/experiments/    A-D (v0.1), E_factorial, T_timing, L_ablation, G_nonpriv, G_search, R_reuse (v0.2)
src/contact_state_reuse/
  envs/                 ContactInsertionEnv, domains, sensor models, labels, scripted demonstrators
  data/                 generation, schema validation, loading, normalisation, episode tensors
  models/               reactive, sequence and predictive models; options for controls
  training/             losses (each logged, gradient norms per term) and the training loop
  evaluation/           rollouts, force interventions, metrics and crossed bootstrap, experiments
  analysis/             latents, probes, CKA, reuse score, reference representations, ablation, plots
scripts/                entry points (run_experiment, run_revision.sh, summarize_*, make_*_figures, videos)
results/revision/       v0.2 results; results/medium/ v0.1 results
docs/whitepaper/        white paper (v0.2 and v0.1), response to reviewers, figures, PDF builder
tests/                  pytest suite
```

## References

Full list in the white paper and in [docs/references.bib](docs/references.bib). Closest work: FoAR (He et al., 2025, RA-L), Force Policy (Fang et al., 2026, RSS), PhaForce (Wang et al., 2026, preprint), ForceVLA (Yu et al., 2025, NeurIPS), TacForeSight (Zang et al., 2026, preprint), Lee et al. (2019, ICRA); methods: CKA (Kornblith et al., 2019), DART (Laskey et al., 2017), the pigeonhole bootstrap (Owen, 2007), and Agarwal et al. (2021) on reporting uncertainty over training runs.

## Author and licence

Maria (Anh Duong Vo). MIT licence.
