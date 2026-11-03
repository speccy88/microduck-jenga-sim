# Local RTX 2060 SUPER Jenga training

This is a simulation experiment, not a verified Jenga policy or a physical robot controller. It uses the existing placement-controlled 61-observation, 14-action Vertical3 task and the unchanged fast-strike success thresholds. The training process does not connect to a robot or cloud provider.

Restore the meshes and pretrained balance checkpoint with the repository's hash-verified `tools/restore_assets.py`; the weights remain subject to their upstream distribution terms. Use the project's locked Python 3.12 environment. A WSL2 test used PyTorch 2.9.1 with CUDA 12.8, mjlab 1.3.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1 and Warp 1.12.0 on an RTX 2060 SUPER. Do not change the MuJoCo version independently of the lock.

From this directory, launch the local runner with a **new output directory** for each run:

```sh
python scripts/local_jenga.py smoke --output /path/to/new/smoke --seconds 900
python scripts/local_jenga.py train --envs 64 --iterations 50 --seconds 900 \
  --resume --checkpoint /path/to/smoke/model_4.pt --output /path/to/new/pilot
```

The smoke uses 32 environments and five PPO iterations, warming the actor and normalizer from the official balance checkpoint while initializing a new critic. The optional `--resume` instead loads the full same-task state. Each run writes a log, sampled GPU/CPU/RAM data and an atomic status JSON; checkpoints are saved in `logs/rsl_rl/`. The timeout terminates the local process group and preserves completed checkpoint files. The runner currently supports Linux/WSL and caps environment counts at 128. Monitor memory, temperature, learning and failures before increasing duration.

The first 32-environment smoke completed five iterations on the RTX 2060 SUPER. The final checkpoint contained finite tensors, and actor tensors changed during PPO. The logged last-iteration throughput was about 166 environment steps/s; sampled GPU memory peaked at 307 MiB and temperature at 46 C, with compilation included in the 171 s wall time. This smoke had no Jenga successes and is not evidence of a trained policy. Sampling may miss the true peak. Contact and head-speed readings in the original PPO environment are sampled at 50 Hz; short impacts may be missed.

A 64-environment pilot resumed the smoke checkpoint for 50 further iterations and completed in 282 seconds including startup. Observed training throughput was roughly 280–377 environment steps/s; sampled GPU memory peaked at 441 MiB. Three diagnostic seeds (9100–9102), evaluated individually on checkpoint 53, all ran the full six seconds without head contact, top-block separation, robot falls, wrong contacts or base failures. This 0/3 diagnostic is not a final held-out evaluation or a convergence estimate. The initial standing behavior is a local optimum candidate; investigate geometry, exploration and the existing sparse strike reward before choosing changes. Do not present checkpoint 53 as a successful policy.

The nominal 20 cm Vertical3 reset places the mouth site approximately 118 mm from the reward's near-face target. A CPU-only grid of 10,000 neck-pitch/head-pitch combinations within the model's joint limits, with the trunk fixed, found a nearest mouth-site position 46.7 mm from that point. This is a static position calculation, not a collision or dynamic reach test; the head geometry can extend beyond the mouth site, and trunk motion can change reach. Together with sparse observed contact, it motivates checking approach exploration and balance, not declaring the task physically impossible. A separate continuation of 200 PPO iterations resumes checkpoint 53 without changing rewards, physics or success thresholds.

That 200-iteration continuation completed in 959 seconds at 64 environments, saving checkpoint 252 and intermediate checkpoints. Sampled GPU usage peaked at 441 MiB and 60 °C. Its checkpoint contained finite tensors and changed actor-network weights. Replaying checkpoint 252 on diagnostic seeds 9100–9102 produced 0/3 successes and no 50 Hz sampled head contacts; all three ran the full six seconds without falls or base failures. These are the same diagnostic seeds used for checkpoint 53, not a final held-out battery. Brief impacts may be missed. No successful Jenga policy was established by this experiment.

For diagnosing exploration, the saved configuration uses gamma 0.99 and scales rewards by the 20 ms step. The approach term is the discounted change of a bounded Gaussian potential in [0, 1], weighted by 2; its discounted cumulative positive contribution before failure is consequently bounded by about 2 reward units. A disqualifying failure costs 25 units before discount, whereas push progress is gated on a qualifying first head strike. This imbalance and the observed no-contact rollouts are evidence for testing a controlled exploration or curriculum ablation, not proof that reward weights alone caused the stall. Keep the original success and failure criteria in all comparisons.

The local launcher accepts an explicitly recorded `--tower-offset` from 0.14 to 0.20 m for a future staged-placement experiment. The default remains 0.20 m. An experiment started closer is a different training distribution; evaluate separately at the original 0.20 m before claiming transfer. Reset geometry, contacts and balance must be checked for the chosen offset before a long run. This option does not change rewards or success thresholds.

The recorder also accepts `--tower-offset` and records it in the manifest. Use `--tower-offset 0.14` for an in-distribution close-start evaluation and a separate output directory with the default 0.20 m for the original-placement transfer check. Do not compare those as though they were the same conditions.

A static nominal-keyframe check found no robot/block contact at offsets 0.135, 0.140 and 0.145 m, or 0.195, 0.200 and 0.205 m. At 0.140 m, the initial mouth-to-target distance is 62.6 mm. These checks cover only longitudinal extremes with the nominal pose; randomized lateral placement, joint poses and GPU reset dynamics still need verification.

A further CPU approximation sampled 200 static resets each at 0.14 m and 0.20 m, varying robot yaw, all 14 actuator positions by ±0.02 rad, and tower placement by ±5 mm on each horizontal axis. None had a detected robot/block contact. This does not exercise robot tilt, domain randomization or GPU stepping, and is not a balance or dynamic-reach test.

A separate 200-reset sample per distance also varied robot roll and pitch by up to one degree; it likewise found no robot/block contact. These samples approximate, but do not fully reproduce, the training reset distribution and do not establish dynamic behavior.

An explicit 14 cm Vertical3 smoke resumed the original smoke checkpoint for five iterations with 32 environments and saved its configuration confirming `offset_x: 0.14` and 5 mm position noise. It completed in 56 seconds; sampled peak GPU memory was 307 MiB and 59 °C. Its actor-network tensors changed and all checked checkpoint tensors were finite. Training logs included head-contact and lower-block-failure events but no qualifying fast strike or success. These logged metrics are not held-out episode rates. The subsequent bounded 100-iteration pilot resumed that same-task checkpoint.

The 100-iteration close-start pilot subsequently completed in 520 seconds, with a sampled peak of 441 MiB and 61 °C. Checkpoint 107 had finite tensors and changed actor MLP weights. On diagnostic seeds 9100–9102, the checkpoint scored 0/3 at 14 cm (one forbidden-contact episode) and 0/3 at 20 cm (no sampled head contacts or failures). These are small comparison samples and not final held-out performance estimates; 50 Hz sampling can miss brief contacts. No successful PPO policy has been demonstrated by this ablation.

The local launcher exposes `--approach-weight` for a controlled exploration ablation. The default remains the original 2.0; an explicit value is recorded in `run.json` and passed to the training configuration. The 20.0 run started from the same close-start checkpoint as the 100-iteration baseline and changed only this approach shaping weight. The motivation was sparse contacts and the bounded approach return relative to failure cost described above. This was a training-reward change, not a change to physics, observations, or success and failure criteria.

The 64-environment, 100-iteration ablation completed in 594 seconds with mean logged throughput 295 environment steps/s, sampled peak VRAM 441 MiB and sampled peak temperature 60 °C. The resulting checkpoint 107 has finite checked tensors and changed actor MLP weights; its SHA-256 is `699203bd7f6389132c524153f900fc3939440b8776f2453dd5d3b3d051201a43`. On the same diagnostic seeds 9100–9102, it scored 0/3 at both 14 cm and the original 20 cm. At each distance seed 9101 had sampled head contact and lower-block base failure (maximum displacement approximately 9.3 mm); the other two had no sampled head contact or success. No robot falls or wrong-contact flags were recorded. At 20 cm the recorded first-contact forward speed for the failed case was 0.032 m/s, below the 0.25 m/s criterion; the PPO measurements are 50 Hz samples and can miss short impacts. These reused diagnostic seeds are not a held-out performance estimate. More approach reward has not established a successful policy or safe transfer.

### Further bounded continuation, 2026-11-03

A further 300 PPO updates resumed the approach-weight-20 checkpoint 107 at the same 14 cm placement, with the same physics, rewards, actor inputs and success thresholds. The environment count increased from 64 to 128, doubling the per-update rollout from 1,536 to 3,072 environment steps. This continuation therefore changes both training duration and rollout batch size; it is not an isolated test of duration alone. It added 921,600 environment steps and completed normally in 1560.787 seconds. Mean logged throughput was 646.0 environment steps/s; sampled GPU memory peaked at 655 MiB and temperature at 60 C.

The last saved checkpoint is iteration 406, SHA-256 `666d14749feb5290df750b48b86d58cf6f7924b6588b06e30b64937bb0b133c7`. Its checked tensors contain no NaN or infinity, and actor-network weights changed relative to checkpoint 107. Checkpoints remain local. The following are complete deterministic-policy diagnostics of this last checkpoint, not a ranking of every saved checkpoint:

| Placement | Reset seeds | Successes | Head contacts | Base failures | Robot falls | Wrong contacts |
|---|---|---:|---:|---:|---:|---:|
| 0.14 m | 9100–9102 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 0.14 m | 9200–9202 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 0.20 m | 9100–9102 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 0.20 m | 9200–9202 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |

The original three reset seeds were reused for comparison; 9200–9202 are additional diagnostic resets. Both conditions retained a fixed startup randomization seed of 41. None of these six reset seeds is final held-out validation. Contact metrics are sampled at 50 Hz and can miss brief impacts. The 20 cm condition is a separate transfer check from the 14 cm training distribution.

This continuation has not established a successful three-block PPO policy. No successful-policy ONNX export or physical robot validation is claimed.

Keep full logs and checkpoints locally. Selection evaluations and held-out evaluations must use separate seeds. A successful release requires reproducible task success, complete failure reporting, verified licensing, and honest simulation-only documentation.
