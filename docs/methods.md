# Methods

This document defines what each experiment does and how each quantity is computed. Code references point to `src/contact_state_reuse/`. Sections 1 to 6 describe the shared machinery and the v0.1 experiments A to D; the v0.2 experiments are summarised directly below and specified in full in the white paper (Section 4).

## v0.2 (revision) at a glance

* **One network.** Every condition of E, G1 and G2 is `PredictiveContactWorldModel` with a GRU of 66 units (61,469 parameters); behaviour cloning is the same network with both prediction weights at 0. Controls: `lambda_recon`, `aux_stopgrad`, `aux_core`, `use_prev_action`, `force_transform` (`models/common.py`).
* **Identical data.** `TrainConfig.episode_filter: success` trains every loss on the successful episodes; failure exposure is a separate condition (`episode_filter: all`).
* **Fault-aware training.** `TrainConfig.force_aug` appends a copy of the training episodes whose force *input* carries the sensor-fault model; prediction targets keep the original readings (`training/trainer.add_force_corruption`).
* **Gradient norms.** The trainer logs the gradient norm of each weighted loss term on the shared encoder and GRU (`grad_<term>` in the history).
* **Force metrics.** `peak_force`: maximum over physics steps of the magnitude of the summed planar contact force; `peak_force_sum`: maximum of the sum of individual contact-force magnitudes (no cancellation).
* **Interventions (T).** `ForceCorruption` modes `first_contact_window`, `onset_offset_window` (offset after onset), `onset_only`, `hold_onset`, `random_window`, `zero`, `scale`; each episode records `masked_steps` and `masked_force`. `run_episodes(..., compliance_override=c)` fixes the compliance command.
* **Behaviour metrics.** Commanded compliance before and after first contact, peg speed before contact, time to first contact, episode length.
* **Demonstrators and second task.** `ExpertConfig(privileged=False)` estimates the slot pose online from the observation; `search=True` adds hole search; the `search` domain offsets the pose estimate by 5 to 9 mm.
* **Statistics.** Crossed seed x episode bootstrap (`evaluation/metrics.py`: `crossed_bootstrap`, `seed_episode_summary`, `paired_crossed`).
* **Representation analysis (R).** Held-out-pair probes (`analysis/probes.paired_heldout_probe`), reference representations (`analysis/latents.reference_latents`: raw observations, proprioception, a progress clock) and an untrained network; reuse null quantiles and sensitivity variants; jackknife intervals (`analysis/reuse.py`); adaptation as gain over zero-shot (`evaluation/revision.py`).

## 1. Task and simulation

* **Mechanism.** A 20 mm x 60 mm peg in the x-z plane with three joints (tip x, tip z, rotation about y through the tip), each driven by a MuJoCo actuator acting as a PD controller: `tau = k (setpoint - q) - c qdot` with `k = kp * (0.25 + 0.75 * compliance)`, `c = 2 * 0.8 * sqrt(k * inertia)`. Physics runs at 500 Hz; the policy acts at 20 Hz.
* **Action.** `[dx, dz, dtheta, compliance] in [-1, 1]^4`; the first three move the set-point by up to 4 mm, 4 mm and 0.03 rad per step; the set-point may not lead the joint by more than 12 mm / 0.15 rad, which caps the force a policy can apply by pushing.
* **Slot.** 40 mm deep, 3 mm chamfer, clearance and tilt randomised. Success: tip at least 25 mm below the entrance and within the clearance laterally.
* **Termination.** Success; jam (in contact for 30 consecutive steps with less than 0.5 mm depth gain and mean force above 2 N); damage (contact force above 40 N). Truncation after 160 steps (8 s).
* **Reward.** `10 * delta(progress) + 10 * success - 0.5 * max(0, peak_force - 10 N) / 10 N - 5 * (jam or damage) - 0.01`, where progress is minus the distance to the entrance above the slot and depth minus lateral error inside it. The reward is logged for completeness; the models in this repository are trained by imitation and auxiliary prediction, not by reinforcement learning.

### Force signal

`ContactInsertionEnv.contact_wrench` sums `mj_contactForce` over all contacts involving the peg, rotates each contact force from the contact frame to the world frame, flips its sign when the peg is the first geom (MuJoCo reports the force of geom1 on geom2), and computes the torque about the peg tip. The observed force is this wrench after the sensor model (`envs/sensors.py`): delay buffer, fixed bias, Gaussian noise, random dropout. This is an idealised external contact wrench. A wrist force/torque sensor would also measure tool inertia and gravity; motor-current sensing would also contain friction and transmission effects.

### Analysis labels

`envs/labels.py` assigns one of six modes per step, in priority order: success, jam or failure, first contact (4 steps from the first contact onset), insertion (tip deeper than the chamfer), alignment (in contact after first contact), approach. These rules are a measurement convention.

## 2. Data

Scripted expert with privileged state, DART noise (sigma 0.1 on motion actions, compliance not perturbed) and a fraction `p_perturbed` of perturbed demonstrations (large aim error, stiff pushing, uncorrected angle error) that produce jams and high-force episodes. Behaviour cloning uses successful episodes only (`bc_on_failures: false`). In v0.1, predictive losses used all episodes; in v0.2 every loss uses the successful episodes (`episode_filter: success`) unless a condition explicitly includes failures.

Pairing: episode *i* of every dataset generated with seed *s* uses environment seed `derive_seed(s, i)`, independent of the domain. Parameters are drawn in a fixed order, so the source and shifted datasets share geometry, start pose and noise streams and differ only in shifted parameters.

## 3. Models and losses

See `models/policies.py`. The total loss is

```
L = L_action + lambda_obs * L_obs + lambda_force * L_force + lambda_contact * L_contact
```

* `L_action`: mean squared error between predicted and clean expert actions, on steps with an action from successful episodes.
* `L_obs`: mean squared error of the standardised next observation (exteroception and proprioception).
* `L_force`: Gaussian negative log-likelihood of the standardised force at t+1..t+5, with predicted mean and log-variance (clamped to [-8, 6]).
* `L_contact`: cross-entropy of the contact mode at t+3. Only active with `contact_supervision: true`.

Every term is computed and logged on validation data even when its weight is 0, so learning curves can be compared.

Matched capacity: `build_model` searches the hidden width so the total parameter count is closest to `param_budget`.

## 4. Evaluation

* Evaluation seeds `derive_seed(seed, "eval", i)` never coincide with data seeds.
* Metrics per episode: success, return, peak force (maximum over all physics steps), contact impulse (sum of force x 0.05 s over in-contact control steps), force violation (peak > 10 N), time to success, jam or failure, first-contact step.
* Summaries: mean and 95 % percentile bootstrap interval over episodes (2000 resamples); seed-level means and standard deviations in `seed_level`; paired differences over matched (seed, domain, episode).
* First-contact prediction (world models): for steps before first contact, score = predicted future force magnitude (max over the 5-step horizon, in newtons), label = first contact within the next 3 steps; AUROC over all such steps.

## 5. Experiments

**A. Modality ablation.** Reactive, sequence and world-model policies, each with extero + proprio vs extero + proprio + force, same parameter budget, data and updates. The world model without force also has no force prediction target, so "without force" means no force anywhere. Evaluated on all domains.

**B. Objective ablation.** Sequence BC; world model with next-observation prediction; with next-observation and future-force prediction; and the simulator-supervised auxiliary variant.

**C. Force timing.** Force readings are set to 0 for `window` = 6 steps (0.3 s) starting at the step where the simulator registers first contact (the harness uses privileged timing only to schedule the intervention), or for 6 steps starting at a uniformly random step in [0, 80), or for the whole episode. Random windows that happen to overlap the first-contact window are flagged; results are also reported without them (`results_random_nonoverlap.csv`). Comparison: paired difference first-contact window minus random window.

**D. Representation reuse.** For each model and seed: pre-train on source; extract latents on a paired analysis set (expert trajectories without perturbations) in source and in the shifted domain at each severity; compute the reuse score and cross-domain probe; evaluate zero-shot; fine-tune from the pre-trained weights on `budget` shifted-domain demonstrations for a fixed number of updates at a lower learning rate; evaluate again.

## 6. Representation analysis

**Latents.** For sequence and world models, z_t is the output of the temporal core; for the reactive model, the fused encoder output. Latents are extracted by running the frozen model over recorded expert trajectories (teacher forcing), so every model is analysed on the same behaviour.

**Probes.** Multinomial logistic regression with standardisation and balanced class weights. Within-domain accuracy uses 5-fold `GroupKFold` with episode ids as groups. Cross-domain accuracy: train on all source rows, test on all shifted rows. Metric: balanced accuracy; chance is 1 / number of classes present.

**Linear CKA.** For column-centred X (n x p) and Y (n x q) with paired rows,

```
CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
```

**Phase-matched pairing.** For phase p and an episode e that visits p in both domains, the latent rows of e in phase p are resampled to m = 4 points by linear interpolation of the row index, giving blocks X_p, Y_p of shape (E_p, m, H), flattened to (E_p m, H).

**Permutation null.** The episode order of Y_p is permuted by a derangement (no episode keeps its partner), 50 to 100 times. This keeps the phase content and within-episode structure and breaks only the correspondence between the same physical situation in the two domains. The null therefore controls for the similarity that any two sets of same-phase latents share.

**Reuse score.**

```
R_p = (CKA(X_p, Y_p) - mean_null_p) / (1 - mean_null_p)
Reuse = mean over phases p in {approach, first contact, alignment, insertion} with E_p >= 5 of R_p
```

R_p = 0: a situation's shifted latents are no more similar to its source latents than to those of a random other episode in the same phase. R_p = 1: identical geometry up to rotation and isotropic scaling. The 95 % interval comes from resampling episodes within each phase (200 to 500 resamples) with the null mean held fixed.

**Adaptation efficiency.** Success as a function of fine-tuning budget b in {0, b_1, ..., b_K}; efficiency = trapezoidal area under this curve divided by b_K - b_0 (mean success over the budget range). Peak-force change = mean peak force at the largest budget minus zero-shot.

**Correlation.** Spearman rho between reuse and adaptation efficiency (and peak-force change) across (model, seed, severity) points, with a percentile bootstrap interval and a two-sided permutation p-value. Points share models and seeds, so they are not independent, and the interval is optimistic. This is an association across conditions.

**Latent feature ablation.** Rank latent dimensions by the summed absolute coefficients of the source contact-mode probe ("contact"), by summed absolute ridge coefficients predicting the expert action ("movement"), or at random (3 draws). Clamp the top k = 25 % of dimensions to their mean over the source analysis set, in the readout and in the GRU state, during closed-loop control, and compare success and force metrics. This is a feature ablation on a trained network, not a lesion and not a causal claim about units.

**Visualisation.** PCA (fit on all rows of one domain) with trajectories coloured by contact mode, force and outcome; event-aligned latent change around first contact. These are descriptive.
