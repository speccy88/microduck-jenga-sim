#!/usr/bin/env python3
"""Build larger-tower README GIF excerpts from preserved recordings, without simulation."""
import argparse
import json

from build_readme_media import MEDIA, capture, probe, run, sha256

PALETTE = (
    ",split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
    "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
)
CLIPS = {
    "tower_gapped_stack": dict(
        sha256="a0fe57166dfa4ff04b67e914cd6f0dc2cd2d41f8dc03a73ed29fdf86a9b054ea",
        source_size=[960, 544], source_frames=500, source_duration_s=20,
        start_s=2, end_s=12, crop_xywh=[20, 80, 800, 400], output_size=[640, 320]),
    "tower_open_portal": dict(
        sha256="818f086186e2d3279fb5327425a410b914bbef7573744b4bff992f37a28b3cda",
        source_size=[960, 544], source_frames=500, source_duration_s=20,
        start_s=2, end_s=12, crop_xywh=[20, 80, 800, 400], output_size=[640, 320]),
    "tower_36cm_scaled": dict(
        sha256="35a06214df789dc287c2c5fef6a834ebe01ef0410b4281d551a0750e9faa37c3",
        source_size=[1280, 720], source_frames=700, source_duration_s=28,
        start_s=10, end_s=22, crop_xywh=[180, 160, 600, 440], output_size=[600, 440]),
}


def filter_for(clip):
    x, y, w, h = clip["crop_xywh"]
    ow, oh = clip["output_size"]
    return (f"trim=start={clip['start_s']}:end={clip['end_s']},setpts=PTS-STARTPTS,"
            f"fps=15,crop={w}:{h}:{x}:{y},scale={ow}:{oh}:flags=lanczos" + PALETTE)


def validate(ffmpeg, ffprobe):
    manifest = json.loads((MEDIA / "tower-media.json").read_text(encoding="utf-8"))
    assert sha256(MEDIA / "tower-experiments.json") == manifest["evidence_sha256"]
    for stem, clip in CLIPS.items():
        source = MEDIA / f"{stem}.mp4"
        assert sha256(source) == clip["sha256"], f"Source hash changed: {source.name}"
        info = probe(ffprobe, source)
        assert [info["width"], info["height"]] == clip["source_size"], info
        assert info["frames"] == clip["source_frames"] and info["frame_rate"] == "25/1", info
        assert abs(info["duration_s"] - clip["source_duration_s"]) < .001, info
        output = MEDIA / f"{stem}.gif"
        assert sha256(output) == manifest["outputs"][output.name]["sha256"]
        run([ffmpeg, "-v", "error", "-i", output, "-f", "null", "-"])
        actual = probe(ffprobe, output)
        duration = clip["end_s"] - clip["start_s"]
        assert [actual["width"], actual["height"]] == clip["output_size"], actual
        assert actual["frames"] == duration * 15, actual
        assert abs(actual["duration_s"] - duration) < .03, actual
        assert b"NETSCAPE2.0\x03\x01\x00\x00\x00" in output.read_bytes()
        assert output.stat().st_size < 3_000_000, "GIF exceeds 3 MB budget"
        print(f"PASS {output.name}: {actual['frames']} frames, {actual['duration_s']:.2f}s, {output.stat().st_size:,} bytes")
    print("PASS tower source/output/evidence hashes, metadata, full GIF decode, timing and infinite loops")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate(args.ffmpeg, args.ffprobe)
        return
    sources, outputs = {}, {}
    for stem, clip in CLIPS.items():
        source, output = MEDIA / f"{stem}.mp4", MEDIA / f"{stem}.gif"
        if sha256(source) != clip["sha256"]:
            raise ValueError(f"Source hash mismatch: {source.name}")
        settings = dict(filter_complex=filter_for(clip), loop=0, gifflags="+transdiff", audio=False)
        run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", source,
             "-filter_complex", settings["filter_complex"], "-an", "-loop", "0",
             "-gifflags", "+transdiff", output])
        sources[source.name] = dict(sha256=clip["sha256"], **probe(args.ffprobe, source))
        outputs[output.name] = dict(sha256=sha256(output), bytes=output.stat().st_size,
                                   excerpt_source_seconds=[clip["start_s"], clip["end_s"]],
                                   settings=settings, **probe(args.ffprobe, output))
    manifest = dict(sources=sources, outputs=outputs,
                    evidence_sha256=sha256(MEDIA / "tower-experiments.json"),
                    tools=dict(ffmpeg=capture([args.ffmpeg, "-version"]).splitlines()[0],
                               ffprobe=capture([args.ffprobe, "-version"]).splitlines()[0]),
                    time_scaling="none; contiguous excerpts at recorded speed")
    (MEDIA / "tower-media.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validate(args.ffmpeg, args.ffprobe)


if __name__ == "__main__":
    main()
