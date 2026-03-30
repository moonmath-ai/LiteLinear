#!/usr/bin/env python3
"""
Generate first-frame JPG thumbnails from MP4s for README previews.
Usage: python extras/gen_video_thumbs.py
Requires: ffmpeg on PATH, or opencv-python (cv2).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed size for all thumbnails (aligns in README)
THUMB_W, THUMB_H = 320, 180

# (directory relative to repo root, thumb subdir name)
DIRS = [
    (REPO_ROOT / "docs" / "assets", "thumbs"),
    (REPO_ROOT / "docs" / "ltx0.9.8_metrics", "thumbs"),
]


def gen_thumb_ffmpeg(mp4: Path, jpg: Path) -> bool:
    """Extract first frame scaled/padded to THUMB_W x THUMB_H."""
    jpg.parent.mkdir(parents=True, exist_ok=True)
    # scale to fit inside THUMB_WxTHUMB_H, then pad to exact size (centered)
    vf = f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=decrease,pad={THUMB_W}:{THUMB_H}:(ow-iw)/2:(oh-ih)/2"
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(mp4),
            "-vframes", "1",
            "-vf", vf,
            "-q:v", "3",
            str(jpg),
        ],
        capture_output=True,
        timeout=30,
    )
    return r.returncode == 0


def gen_thumb_cv2(mp4: Path, jpg: Path) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    jpg.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    h, w = frame.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    nw, nh = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    pad = THUMB_W - nw, THUMB_H - nh
    top, left = pad[1] // 2, pad[0] // 2
    frame = cv2.copyMakeBorder(frame, top, pad[1] - top, left, pad[0] - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.imwrite(str(jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    for dir_path, thumb_subdir in DIRS:
        if not dir_path.is_dir():
            continue
        thumb_dir = dir_path / thumb_subdir
        thumb_dir.mkdir(parents=True, exist_ok=True)
        for mp4 in sorted(dir_path.glob("*.mp4")):
            # validate_r64_..._Acharminganimatedsceneofafluff.calib.mp4 -> ...calib.jpg
            base = mp4.stem  # no .mp4
            jpg = thumb_dir / f"{base}.jpg"
            if jpg.exists() and jpg.stat().st_mtime >= mp4.stat().st_mtime:
                print(f"skip (up to date): {jpg.relative_to(REPO_ROOT)}")
                continue
            if gen_thumb_ffmpeg(mp4, jpg):
                print(f"ffmpeg: {jpg.relative_to(REPO_ROOT)}")
            elif gen_thumb_cv2(mp4, jpg):
                print(f"cv2: {jpg.relative_to(REPO_ROOT)}")
            else:
                print(f"FAIL: {mp4.relative_to(REPO_ROOT)}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
