# Data schema (version 1)

A dataset is two files written by `scripts/generate_dataset.py` (`data/generate.py`):

```
<name>.npz        arrays (compressed NumPy)
<name>.meta.json  metadata
```

`data.load_dataset(path)` loads both and runs `data.schema.validate`, which checks presence, shapes, finite values, action range, label range, episode grouping and the final-row convention, and raises `SchemaError` otherwise.

## Rows

An episode with T actions contributes T + 1 rows. Row t holds observation o_t, the analysis labels describing the state at o_t, and the action taken from o_t. The final row of each episode holds the final observation and labels, with `has_action = False` and a zero action, so terminal labels (success, jam) are present.

| field | shape | dtype | policy visible | description |
|---|---|---|---|---|
| `obs` | [N, D] | float32 | yes | observation; layout in `meta.obs_layout` (default D = 14: extero 0:4, proprio 4:11, force 11:14) |
| `action` | [N, 4] | float32 | yes (as previous action) | executed action (expert + DART noise), 0 on final rows |
| `expert_action` | [N, 4] | float32 | no | clean expert action, the behaviour-cloning target |
| `reward` | [N] | float32 | no | reward after `action` |
| `has_action` | [N] | bool | no | False on final rows |
| `episode_id` | [N] | int32 | no | 0..E-1, rows grouped by episode in order |
| `t` | [N] | int16 | no | control step |
| `contact_mode` | [N] | int8 | **no** | 0 approach, 1 first_contact, 2 alignment, 3 insertion, 4 success, 5 jam_or_failure |
| `first_contact` | [N] | bool | **no** | True on the first step with contact |
| `success` | [N] | bool | **no** | goal depth reached at this row |
| `jam` | [N] | bool | **no** | jam or damage detected at this row |
| `insertion_depth` | [N] | float32 | **no** | peg tip depth below the slot entrance [m] |
| `contact_force_norm` | [N] | float32 | **no** | true contact force magnitude [N] |
| `peak_force` | [N] | float32 | **no** | true peak force over the preceding control interval [N] |
| `wrench_true` | [N, 3] | float32 | **no** | true planar wrench (Fx, Fz [N], Ty [N m]) |

## Episodes

| field | shape | description |
|---|---|---|
| `ep_domain` | [E] str | concrete domain (for `mixed`, the domain drawn for that episode) |
| `ep_variant` | [E] str | expert variant: nominal, large_aim, stiff_push, angle_error |
| `ep_success` | [E] bool | outcome |
| `ep_jam` | [E] bool | ended in jam or damage |
| `ep_length` | [E] int32 | number of actions T |
| `ep_seed` | [E] int64 | environment seed |

## Metadata (`.meta.json`)

`schema_version`, `package_version`, `code_version` (git hash or "unknown"), `dataset` (the `DatasetConfig`), `env_config`, `expert_config`, `obs_layout`, `n_rows`, `n_episodes`, `success_rate`, `jam_rate`, `variants` (counts) and `episode_params` (the randomised parameters of every episode).

## Conventions

* Units: metres, radians, newtons, seconds; control step 0.05 s.
* Fields marked "no" in the policy-visible column must never be given to a policy as input. `contact_mode` may be used as a training target only in configurations with `contact_supervision: true`.
* Datasets generated with the same seed in different domains are paired episode by episode (see `docs/methods.md`).
