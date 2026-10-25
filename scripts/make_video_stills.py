"""Still images for the PDF: a mid-episode frame and the final frame of each video, stacked.

python scripts/make_video_stills.py assets/videos/v2_*.mp4 --out docs/whitepaper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", default="docs/whitepaper/figures")
    p.add_argument("--hold", type=int, default=20, help="frames held at the end of each video")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        frames = imageio.mimread(v, memtest=False)
        n = len(frames) - a.hold
        still = np.concatenate([frames[n // 2], np.full((8, frames[0].shape[1], 3), 252, np.uint8), frames[-1]], 0)
        dst = out / f"video_{Path(v).stem}.png"
        imageio.imwrite(dst, still)
        print("wrote", dst)


if __name__ == "__main__":
    main()
