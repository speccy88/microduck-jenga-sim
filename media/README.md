# Simulation media: attribution and provenance

`cem_2block_seed_61000.mp4` shows a simulated two-block CEM-controller trial, not a three-block PPO policy or a physical robot. Its 301 frames are rendered at 50 frames per second. The top block falls while the lower block and robot remain standing; the matching first-contact head speed was 0.340 m/s.

`cem_2block_seed_61001_failure.mp4` shows a failure from the same two-block controller. The top block falls, but first-contact head speed was 0.244 m/s, below the 0.25 m/s criterion. It also contains 301 frames at 50 frames per second.

The video depicts Pollen Robotics' Microduck 3D models from [microduck_rl](https://github.com/pollen-robotics/microduck_rl), whose model files are licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The recording is shared under CC BY-NC-SA 4.0; the scene and animation were created for this experiment. The source robot meshes are not distributed here. See [LICENSE-MODELS.md](../LICENSE-MODELS.md).

SHA-256: `16664f417208918a04ea867559d25cbf8af8226f18aa8842cb5c79b09f8d16e5`.

Failure-video SHA-256: `8c53be21912bf1506d733b25837871fda197046417bcc575d2f9994bb58541ea`.

## Derived README media

The two CEM GIFs and the footage-based banner are adaptations of those recordings,
shared under the **same CC BY-NC-SA 4.0 license**. Credit **Pollen Robotics** for
the MicroDuck models and [this project](https://github.com/speccy88/microduck-jenga-sim)
for the simulation scene, recordings and README adaptations. Link to
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) and the
[CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
when sharing. The original robot meshes and pretrained weights remain unbundled.

| File | Source and changes | Output |
| :--- | :--- | :--- |
| [Success GIF](cem_2block_seed_61000.gif) | Entire seed 61000 MP4; spatial crop, reduced frame rate and color palette | 500 × 500; 151 frames; 25 fps; 6.04 s; infinite loop |
| [Failure GIF](cem_2block_seed_61001_failure.gif) | Entire seed 61001 MP4; same conversion settings | 500 × 500; 151 frames; 25 fps; 6.04 s; infinite loop |
| [README banner](readme-banner.png) | Seed 61000, zero-based frame 90 (1.80 s); cropped still combined with project typography | 1280 × 400 PNG |

The source clips are 960 × 720, 301 frames at 50 fps (6.02 s). No action was
generated, reordered, interpolated or sped up. The GIFs preserve the full trials;
25 fps frame sampling quantizes the final duration to 6.04 s. Their crop is
`x=260, y=130, width=500, height=500`, retaining the robot, tower and fallen block.
The banner still uses `x=190, y=140, width=600, height=420` before layout.
All positions refer to the source pixels, measured from the top left.

Both trials show a **preassembled two-block tower**. The falling block in the
failed trial is intentional evidence: its measured 0.244 m/s first-contact head
speed misses the 0.25 m/s requirement. See the
[experiment record](../microduck_jenga_faststrike/microduck_jenga/docs/prior_14cm_evidence.md).

## Rebuild and validate

From the repository root, with Python 3.10+, FFmpeg/ffprobe and Node.js/npx on
`PATH`:

```sh
python tools/build_readme_media.py
python tools/build_readme_media.py --check
```

No Python packages, simulation environment, robot meshes, model weights or
training job are needed. `--skip-banner` rebuilds just the GIFs and preserves
the existing PNG. The optional `--ffmpeg`, `--ffprobe` and `--npx` flags accept
explicit executable paths.

The GIF filter is:

```text
fps=25,crop=500:500:260:130,split[a][b];
[a]palettegen=max_colors=128:stats_mode=diff[p];
[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle
```

FFmpeg writes silent GIFs with `-loop 0 -gifflags +transdiff`. The banner is a
still HyperFrames `0.9.17` composition from
[`tools/readme_banner.html`](../tools/readme_banner.html); the builder extracts
the authentic source frame, runs HyperFrames' layout/runtime/contrast check,
then takes a single snapshot. The first build may download the pinned CLI,
Chromium and the Inter font. It does not publish a HyperFrames project.

[`readme-media.json`](readme-media.json) records source and output SHA-256
hashes, exact tool versions, crop/filter settings, dimensions, frame counts,
durations and file sizes. Use those versions for byte-level reproduction;
different FFmpeg builds or browser/font rasterizers can produce different
hashes. The two CEM GIFs together are about **2.4 MiB**; the banner is about 0.13 MiB.

The `--check` command verifies source hashes and metadata, output hashes, a full
decode of every output, GIF dimensions/frame counts/timing, infinite-loop
extensions and a 3 MB limit per GIF. Representative source and generated frames
were also visually inspected to confirm the tower, strike and falling block
remain visible.

## Whole-tower experiments

These are the earlier **whole-tower knockdown** experiments, separate from the
two-block CEM and three-block selective PPO experiments. Their upper control
is scripted camera/ToF approach and push over frozen learned locomotion, with
lightweight learned semantic perception. No new attack policy was trained.
Each clip is one development trial with geometry seed 101 and reset seed 9601;
the clips do not provide held-out success-rate estimates or physical robot tests.

| Checked-in source | Recorded experiment | Observed full-trial outcome |
| :--- | :--- | :--- |
| [tower_gapped_stack.mp4](tower_gapped_stack.mp4) | `controlled_approach_v3 / gapped_stack`: 12 blocks, six layers, 9 cm initial nominal height; 20 s | Contact and scattered blocks; robot upright. Final remaining height 3.507 cm, floor fraction 80%; strict collapse false. |
| [tower_open_portal.mp4](tower_open_portal.mp4) | `controlled_approach_v3 / open_portal`: six blocks, two levels, 18 cm initial nominal height; 20 s | Contact and partial knockdown; robot upright. Final remaining height 3.504 cm, floor fraction 75%; strict collapse false. |
| [tower_36cm_scaled.mp4](tower_36cm_scaled.mp4) | `tall_scaled_primitive_v1`: 72 blocks, 24 layers, 36 cm initial nominal height; 28 s | No robot/tower contact, no collapse, robot upright; final tower height about 35.94 cm. |

All blocks are 75 × 25 × 15 mm. The first two clips retain the original robot
geometry, measured at 27.22 cm in the recorded reference pose. The tall clip
uses a **separate resized simulation profile** measuring 25.00 cm in that pose:
length scale 0.9183326458, unchanged body masses and length-squared inertias.
Height changes with posture; this is not physical calibration or a gait-transfer
validation. The tall overview varies block tints for readability; its geometry,
recorded motion and original head-camera inset are unchanged.

The tall experiment explicitly uses the primitive BOX/BOX collider with
NATIVECCD disabled. It passed a two-second unforced stability check (maximum
vertical displacement 0.629 mm). A different default-collider run collapsed
before robot contact and is **excluded**; it is not the footage shown here.

The complete-collapse check requires more than low debris: at least 80% height
potential loss, at least 90% of initially elevated blocks near the floor,
initially vertical columns lying down, low residual height, block speed below
0.15 m/s and a 0.5 s stable hold. The two contact trials fail that check and
camera inspection remains inconclusive. No accepted victory or victory photo
was recorded. These outcomes must not be combined with the 25/30 CEM result.

### Source lineage and attribution

The two overview files are copied unchanged from
`attack_video_evidence_v3/{gapped_stack,open_portal}_overview.mp4`. They replay
saved GPU qpos/qvel from `controlled_approach_v3`, with no new integration.
The same overview tracks were used in the previously delivered 40 s video
`Microduck_attaque_tours_dernier_essai_simule.mp4` (SHA-256
`db97666394c8691baf77458cfaadfbe84737d31f434527f6e6c5f20294042dc1`),
which also contains the original head-camera images and labels.

The tall source is an unchanged copy of the delivered
`Microduck_25cm_tour_36cm_blocs_7p5x2p5x1p5cm_simulation.mp4`, from
`tall_attack_video_v2`, SHA-256
`35a06214df789dc287c2c5fef6a834ebe01ef0410b4281d551a0750e9faa37c3`.
It replays recorded states from `tall_scaled_primitive_v1` and includes the
original simulated head-camera inset. The GIF crops the overview only.

[`tower-experiments.json`](tower-experiments.json) preserves selected raw
outcome/scale fields, original record names and SHA-256 hashes, trajectory
hashes and delivery lineage. Original experiment files remain unchanged in
the local archive; no checkpoint, weights, robot meshes or simulation runtime
code are added by this gallery update.

These three source recordings and derived GIFs depict **Pollen Robotics'**
MicroDuck models from [microduck_rl](https://github.com/pollen-robotics/microduck_rl).
They are shared under **CC BY-NC-SA 4.0**, with credit to Pollen Robotics for
the models and [this project](https://github.com/speccy88/microduck-jenga-sim)
for the scenes, recordings and GIF adaptations. Retain the credit, source link,
[license link](https://creativecommons.org/licenses/by-nc-sa/4.0/) and modification
notice when sharing. The GIF adaptations trim, crop, reduce frame rate and
quantize colors; they do not generate motion or alter the playback speed.

### Rebuild the larger-tower GIFs

With Python 3.10+ and FFmpeg/ffprobe, from the repository root:

```sh
python tools/build_tower_media.py
python tools/build_tower_media.py --check
```

No Python packages, HyperFrames renderer, robot assets or training environment
are needed for these format conversions. Explicit `--ffmpeg` and `--ffprobe`
paths are supported. The existing CEM/banner builder remains separate.

| GIF | Source interval, at recorded speed | Source crop (x, y, width, height) | GIF dimensions / frame count |
| :--- | :--- | :--- | :--- |
| [Gapped stack](tower_gapped_stack.gif) | [2, 12) s of 20 s | 20, 80, 800, 400 | 640 × 320 / 150 |
| [Open portal](tower_open_portal.gif) | [2, 12) s of 20 s | 20, 80, 800, 400 | 640 × 320 / 150 |
| [36 cm tower](tower_36cm_scaled.gif) | [10, 22) s of 28 s | 180, 160, 600, 440 | 600 × 440 / 180 |

All three use 15 fps, `palettegen=max_colors=96:stats_mode=diff` and
`paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle`, with
`-loop 0 -gifflags +transdiff`. The overview pair is scaled with Lanczos;
the tall crop retains source pixel dimensions. They total about **2.4 MiB**.
All five README GIFs together are about **4.8 MiB**.

[`tower-media.json`](tower-media.json) records complete filter strings, source
and output hashes, source metadata, file sizes and exact conversion versions.
The builder's `--check` validates source/output/evidence hashes, source
metadata, full output decode, frame counts, durations, infinite loops and the
3 MB per-GIF budget. Representative source and generated frames were inspected
to ensure the initial structures, robot and resulting debris remain visible.
