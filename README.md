# Microduck Jenga simulation and training

This placement-controlled Microduck Jenga project includes a three-block PPO training environment and a separate two-block controller experiment. The PPO experiments have not established a successful three-block policy. No physical robot testing has been performed. The robot is initially placed relative to the tower; the actor does not locate it with a camera or ToF sensor. See [local training results](microduck_jenga_faststrike/microduck_jenga/docs/local_rtx_training.md) for measurements and limitations.

A separate simulation uses a cross-entropy-method (CEM) optimized head trajectory over a frozen VelStand balance policy, with ideal simulated bearing. At a controlled 14 cm starting placement and with two free blocks, the historical controller passed 25 of 30 new reset seeds (83.3%; 95% Wilson interval 66.4%–92.7%) under the recorded strict fast-strike and stability criteria. The environment startup seed remained fixed. Two repeated impacts were cross-checked against CPU MuJoCo at matching simulated positions. The candidate source reproduced one success and one failure on the same two reference seeds using a freshly installed locked environment. This is not an end-to-end PPO policy, a camera-based localization system, a three-block result or a physical robot test.

Watch the [two-block CEM success, seed 61000](media/cem_2block_seed_61000.mp4). The [raw evaluation](microduck_jenga_faststrike/microduck_jenga/cem/evaluation_30_resets.json), [summary](microduck_jenga_faststrike/microduck_jenga/cem/evaluation_summary.json), and [provenance and contact checks](microduck_jenga_faststrike/microduck_jenga/docs/prior_14cm_evidence.md) document the experiment and its failures. The video depicts Pollen Robotics' 3D robot models; see [media attribution](media/README.md).

The code is derived from Pollen Robotics microduck_rl and is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Meshes are omitted and must be obtained separately under CC BY-NC-SA 4.0; see [LICENSE-MODELS.md](LICENSE-MODELS.md). Third-party pretrained weights are not bundled. Their own distribution terms must be confirmed.

## Installation and local tests

Use Linux or WSL2 with Python 3.12, a compatible NVIDIA CUDA GPU and `uv`. The lockfile was exercised with mjlab 1.3.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1, Warp 1.12.0 and PyTorch 2.9.1. An RTX 2060 SUPER with 8 GB was used for local experiments; other hardware is unverified. Review the upstream asset and model terms before fetching them.

```sh
python tools/restore_assets.py --plan
# Once terms have been reviewed, restore public, hash-verified upstream assets:
python tools/restore_assets.py --fetch
cd microduck_jenga_faststrike/microduck_jenga
uv sync --locked
CUDA_VISIBLE_DEVICES='' uv run python -m unittest discover -s tests -p test_jenga_preparation.py -v
CUDA_VISIBLE_DEVICES='' uv run python scripts/validate_jenga.py
uv run python scripts/local_jenga.py smoke --output /path/to/new/smoke --seconds 900
# Once the restored balance checkpoint is available, replay the separate CEM controller:
uv run python scripts/evaluate_cem.py --checkpoint cem/checkpoint_100.json --output /path/to/new/cem-eval --first-seed 61000 --count 30
uv run python scripts/summarize_cem.py /path/to/new/cem-eval/results.json
```

The local runner needs the restored VelStand training checkpoint. It saves logs, hardware samples and checkpoints and never starts a paid cloud resource. See `docs/local_rtx_training.md` for the measured results, resume instructions and explicit placement settings. A fresh locked installation passed 22 helper tests, nine CPU physics checks, and a five-iteration 32-environment GPU smoke on the RTX 2060 SUPER. Broader hardware compatibility and third-party model-weight distribution rights remain unverified. The project does not establish physical actuator safety.
