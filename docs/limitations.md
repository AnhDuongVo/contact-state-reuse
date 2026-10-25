# Limitations

## What the simulation does not capture

* **Force sensing.** The force channel is the sum of MuJoCo contact forces on the peg, an idealised external wrench at the contacts. A wrist force/torque sensor also measures tool inertia, gravity and its own compliance. Force estimated from motor currents (as in force-transparent actuators) additionally contains joint friction, gearing losses, temperature drift and calibration error. Results about "force" here are about access to an ideal contact signal plus the noise, bias, delay and dropout the sensor model adds.
* **Contact physics.** MuJoCo uses soft contacts. Force magnitudes depend on solver parameters, contact time constants and the 2 ms step. Peak forces are comparable between conditions in this simulator, not as absolute newtons on hardware.
* **Planar, rigid task.** Three degrees of freedom, rigid peg and slot, no wrist rotation out of plane, no cables or deformable parts, no gripper slip.
* **Exteroception is not vision.** The "camera" is a noisy, biased, occludable estimate of the relative slot pose. It isolates the information a visual estimate provides but not the difficulty of learning it from pixels.
* **Labels are conventions.** Contact modes are produced by fixed rules. Real interactions change continuously and can be in several regimes at once.

## What the experiments cannot show

* **No hardware validation.** Nothing here establishes real-robot transfer.
* **Limited diversity.** One task family and a handful of physical shifts. Reuse across genuinely different tasks is untested.
* **Fixed demonstrator.** All demonstrations come from one scripted expert and its perturbed variants. Policies can only learn strategies that expert shows, and the expert uses privileged state that the policy does not have.
* **Small models and data.** About 60k parameters and hundreds of demonstrations; conclusions may change at foundation-model scale.
* **Correlation, not causation.** The reuse/adaptation analysis relates two measured quantities across conditions that share models and seeds. A positive association would be consistent with reuse helping adaptation but would not show it.
* **Latent feature ablation** clamps dimensions of a trained network. Redundancy means a null effect does not show the information is unused, and a large effect does not identify a mechanism.

## Next steps

1. Real robot data with joint torque or motor-current sensing alongside a wrist sensor.
2. Actuator-current models: simulate motor friction, gearing and calibration error so that "proprioceptive force" is a realistic estimate rather than an ideal measurement.
3. Tactile sensing for local contact geometry.
4. Scalable multimodal pre-training across many tasks and data mixtures, then test reuse on held-out tasks rather than held-out physics.
5. Learned compliance: let the policy's compliance output be studied as part of the adaptation.
