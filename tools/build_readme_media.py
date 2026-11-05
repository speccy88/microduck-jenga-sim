#!/usr/bin/env python3
"""Build README GIFs from the checked-in recordings; never run a simulation.

Python 3.10+, FFmpeg/ffprobe and (for the banner) Node.js/npx are required.
No Python packages are needed. See media/README.md for provenance and commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
SOURCES = {
    "cem_2block_seed_61000.mp4": "16664f417208918a04ea867559d25cbf8af8226f18aa8842cb5c79b09f8d16e5",
    "cem_2block_seed_61001_failure.mp4": "8c53be21912bf1506d733b25837871fda197046417bcc575d2f9994bb58541ea",
}
FILTER = (
    "fps=25,crop=500:500:260:130,split[a][b];"
    "[a]palettegen=max_colors=128:stats_mode=diff[p];"
    "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
)
HYPERFRAMES = "hyperframes@0.9.17"


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def capture(args):
    return run(args, capture_output=True, text=True).stdout


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(ffprobe, path):
    result = json.loads(capture([ffprobe, "-v", "error", "-select_streams", "v:0",
                                "-count_frames", "-show_entries",
                                "stream=width,height,r_frame_rate,nb_read_frames,duration:format=duration",
                                "-of", "json", path]))
    stream = result["streams"][0]
    if path.suffix == ".png":
        return dict(width=int(stream["width"]), height=int(stream["height"]), frames=1)
    duration = result["format"].get("duration")
    if duration is None and path.suffix == ".gif":
        packets = json.loads(capture([ffprobe, "-v", "error", "-select_streams", "v:0",
                                      "-show_packets", "-show_entries", "packet=duration_time",
                                      "-of", "json", path]))
        duration = sum(float(p["duration_time"]) for p in packets["packets"])
    return dict(width=int(stream["width"]), height=int(stream["height"]),
                frames=int(stream["nb_read_frames"]), frame_rate=stream["r_frame_rate"],
                duration_s=round(float(duration), 6) if duration is not None else None)


def validate(ffmpeg, ffprobe):
    """Decode every output, check timing, hashes and the GIF loop extension."""
    manifest = json.loads((MEDIA / "readme-media.json").read_text(encoding="utf-8"))
    for name, expected in SOURCES.items():
        assert sha256(MEDIA / name) == expected, f"Source changed: {name}"
        info = probe(ffprobe, MEDIA / name)
        assert (info["width"], info["height"], info["frames"]) == (960, 720, 301), info
        assert info["frame_rate"] == "50/1", info
        assert abs(info["duration_s"] - 6.02) < 0.001, info
    for name, recorded in manifest["outputs"].items():
        path = MEDIA / name
        assert sha256(path) == recorded["sha256"], f"Output changed: {name}"
        run([ffmpeg, "-v", "error", "-i", path, "-f", "null", "-"])
        if path.suffix == ".gif":
            info = probe(ffprobe, path)
            assert (info["width"], info["height"], info["frames"]) == (500, 500, 151), info
            assert abs(info["duration_s"] - 6.04) < 0.011, info
            assert b"NETSCAPE2.0\x03\x01\x00\x00\x00" in path.read_bytes(), "GIF must loop forever"
            assert path.stat().st_size < 3_000_000, "GIF exceeds the 3 MB budget"
            print(f"PASS {name}: {info['frames']} frames, {info['duration_s']:.2f}s, {path.stat().st_size:,} bytes")
    print("PASS source hashes, source timing, full output decode, output hashes and infinite GIF loops")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--npx", default="npx", help="npx executable; used only to build the banner")
    parser.add_argument("--check", action="store_true", help="Validate existing assets without rebuilding")
    parser.add_argument("--skip-banner", action="store_true", help="Rebuild GIFs only, preserving an existing banner")
    args = parser.parse_args()
    if args.check:
        validate(args.ffmpeg, args.ffprobe)
        return
    for name, expected in SOURCES.items():
        if sha256(MEDIA / name) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    for name in SOURCES:
        run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", MEDIA / name,
             "-filter_complex", FILTER, "-an", "-loop", "0", "-gifflags", "+transdiff",
             MEDIA / name.replace(".mp4", ".gif")])
    if not args.skip_banner:
        with tempfile.TemporaryDirectory(prefix="microduck-readme-") as directory:
            project = Path(directory)
            (project / "assets").mkdir()
            shutil.copyfile(ROOT / "tools" / "readme_banner.html", project / "index.html")
            run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i",
                 MEDIA / "cem_2block_seed_61000.mp4", "-vf", "select=eq(n\\,90),crop=600:420:190:140",
                 "-frames:v", "1", project / "assets" / "strike.png"])
            run([args.npx, "--yes", HYPERFRAMES, "check", project, "--json"])
            output = project / "snapshot"
            run([args.npx, "--yes", HYPERFRAMES, "snapshot", project,
                 "--at", "0", "--no-end", "--output", output, "--json"])
            frames = list(output.rglob("*.png"))
            if len(frames) != 1:
                raise RuntimeError(f"Expected one banner snapshot, got {frames}")
            shutil.copyfile(frames[0], MEDIA / "readme-banner.png")
    outputs = {}
    for name in [n.replace(".mp4", ".gif") for n in SOURCES] + ["readme-banner.png"]:
        path = MEDIA / name
        outputs[name] = dict(sha256=sha256(path), bytes=path.stat().st_size,
                             **probe(args.ffprobe, path))
    manifest = dict(
        sources={name: dict(sha256=digest, **probe(args.ffprobe, MEDIA / name))
                 for name, digest in SOURCES.items()},
        settings=dict(filter_complex=FILTER, loop=0, gifflags="+transdiff", audio=False,
                      source_range="full clip; no time scaling", source_crop_xywh=[260, 130, 500, 500],
                      banner_source_frame=90, banner_crop_xywh=[190, 140, 600, 420],
                      banner_hyperframes=HYPERFRAMES),
        tools=dict(ffmpeg=capture([args.ffmpeg, "-version"]).splitlines()[0],
                   ffprobe=capture([args.ffprobe, "-version"]).splitlines()[0]),
        outputs=outputs,
    )
    (MEDIA / "readme-media.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate(args.ffmpeg, args.ffprobe)


if __name__ == "__main__":
    main()
