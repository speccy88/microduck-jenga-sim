# Simulation media: attribution and provenance

`cem_2block_seed_61000.mp4` shows a simulated two-block CEM-controller trial, not a three-block PPO policy or a physical robot. Its 301 frames are rendered at 50 frames per second. The top block falls while the lower block and robot remain standing; the matching first-contact head speed was 0.340 m/s.

`cem_2block_seed_61001_failure.mp4` shows a failure from the same two-block controller. The top block falls, but first-contact head speed was 0.244 m/s, below the 0.25 m/s criterion. It also contains 301 frames at 50 frames per second.

The video depicts Pollen Robotics' Microduck 3D models from [microduck_rl](https://github.com/pollen-robotics/microduck_rl), whose model files are licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The recording is shared under CC BY-NC-SA 4.0; the scene and animation were created for this experiment. The source robot meshes are not distributed here. See [LICENSE-MODELS.md](../LICENSE-MODELS.md).

SHA-256: `16664f417208918a04ea867559d25cbf8af8226f18aa8842cb5c79b09f8d16e5`.

Failure-video SHA-256: `8c53be21912bf1506d733b25837871fda197046417bcc575d2f9994bb58541ea`.

## Derived README media

The two GIFs and the footage-based banner are adaptations of those recordings,
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
hashes. The two GIFs together are about **2.4 MiB**; the banner is about 0.12 MiB.

The `--check` command verifies source hashes and metadata, output hashes, a full
decode of every output, GIF dimensions/frame counts/timing, infinite-loop
extensions and a 3 MB limit per GIF. Representative source and generated frames
were also visually inspected to confirm the tower, strike and falling block
remain visible.
