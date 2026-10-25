"""Side-by-side rollout videos and force traces for identical episodes.

Both policies face exactly the same episode (same evaluation seed: geometry,
start pose, physics and sensor-noise streams), so differences in the video
come from the policies, not from the episode.

Examples:
    MUJOCO_GL=osmesa python scripts/make_videos.py \
        --a outputs/A_modality_medium/<run>/checkpoints/world_model_EP_seed0.pt \
        --b outputs/A_modality_medium/<run>/checkpoints/world_model_EPF_seed0.pt \
        --domain tight_clearance --episodes 0 1 2
    # same policy with and without force around first contact
    MUJOCO_GL=osmesa python scripts/make_videos.py --a ckpt.pt --b ckpt.pt \
        --corruption-b first_contact_window --labels "intact force" "force zeroed at first contact"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from contact_state_reuse.analysis.plotting import MODE_COLORS, PALETTE  # noqa: E402
from contact_state_reuse.envs import MODE_NAMES, ContactInsertionEnv  # noqa: E402
from contact_state_reuse.evaluation.rollout import ForceCorruption, PolicyAgent, _CorruptionState  # noqa: E402
from contact_state_reuse.models import load_checkpoint  # noqa: E402
from contact_state_reuse.utils.seeding import derive_seed  # noqa: E402

H, W = 360, 480


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rollout(ckpt: str, domain: str, severity: float, seed: int, episode: int, corruption: ForceCorruption):
    model, _, norm, _ = load_checkpoint(ckpt)
    agent = PolicyAgent(model, norm)
    env = ContactInsertionEnv(domain=domain, severity=severity)
    es = derive_seed(seed, "eval", episode)
    obs, info = env.reset(seed=es, options={"domain": domain, "severity": severity})
    cs = _CorruptionState(corruption, np.random.default_rng(derive_seed(es, "corrupt")), env.layout["force"])
    obs = cs.apply(obs, 0, False)
    renderer = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ent = env.entrance()
    cam.lookat[:] = [ent[0], 0.0, ent[1] + 0.012]
    cam.distance, cam.azimuth, cam.elevation = 0.25, 90.0, -10.0
    agent.reset(1)
    frames, infos, forces, observed = [], [], [], []
    done = False
    while True:
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render().copy())
        infos.append(info)
        observed.append(float(np.hypot(*obs[env.layout["force"]][:2])))
        if done:
            break
        act, _ = agent.act(obs[None])
        obs, _, term, trunc, info = env.step(act[0])
        obs = cs.apply(obs, info["t"], info["first_contact"])
        forces.append(info["peak_force"])
        done = term or trunc
    renderer.close()
    outcome = "success" if info["success"] else ("jam / failure" if (info["jam"] or info["damage"]) else "timeout")
    return frames, infos, np.asarray(forces), np.asarray(observed), outcome, env


def annotate(
    frame: np.ndarray,
    title: str,
    color: tuple[int, int, int],
    info: dict,
    observed: float,
    final: str | None,
    limit: float,
    corrupted: bool,
) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill=(252, 252, 251))
    d.rectangle([0, 0, 8, 30], fill=color)
    d.text((14, 6), title, fill=(11, 11, 11), font=_font(15))
    mode = info["contact_mode"]
    mc = tuple(int(MODE_COLORS[mode][i : i + 2], 16) for i in (1, 3, 5))
    d.rectangle([W - 132, 36, W - 8, 58], fill=mc)
    d.text((W - 126, 39), MODE_NAMES[mode].replace("_", " "), fill=(11, 11, 11), font=_font(13))
    f = info["peak_force"]
    frac = min(1.0, f / (2 * limit))
    d.rectangle([10, H - 24, W - 10, H - 12], outline=(180, 180, 175))
    d.rectangle([10, H - 24, 10 + int((W - 20) * frac), H - 12], fill=(208, 59, 59) if f > limit else (42, 120, 214))
    x_lim = 10 + int((W - 20) * 0.5)
    d.line([x_lim, H - 28, x_lim, H - 8], fill=(208, 59, 59), width=2)
    txt = f"t={info['t'] * 0.05:4.2f}s  true |F|={f:5.1f} N  observed |F|={observed:4.1f} N"
    if corrupted:
        txt += "  (force input zeroed)"
    d.text((12, H - 46), txt, fill=(11, 11, 11), font=_font(12))
    d.text((12, 38), f"depth {1e3 * info['insertion_depth']:5.1f} mm", fill=(11, 11, 11), font=_font(12))
    if final:
        d.text((12, 58), final, fill=(0, 131, 0) if final == "success" else (208, 59, 59), font=_font(16))
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint shown on the left")
    p.add_argument("--b", required=True, help="checkpoint shown on the right")
    p.add_argument("--labels", nargs=2, default=None)
    p.add_argument("--domain", default="tight_clearance")
    p.add_argument("--severity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123, help="evaluation seed")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--corruption-a", default="none")
    p.add_argument("--corruption-b", default="none")
    p.add_argument("--out", default="assets/videos")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = a.labels or [Path(a.a).stem, Path(a.b).stem]
    specs = [(a.a, labels[0], ForceCorruption(a.corruption_a)), (a.b, labels[1], ForceCorruption(a.corruption_b))]
    for ep in a.episodes:
        runs = [rollout(c, a.domain, a.severity, a.seed, ep, cor) for c, _, cor in specs]
        limit = runs[0][5].cfg.safe_force
        n = max(len(r[0]) for r in runs)
        video = []
        for k in range(n):
            panels = []
            for j, ((frames, infos, _f, observed, outcome, _e), (_, lab, cor)) in enumerate(zip(runs, specs)):
                i = min(k, len(frames) - 1)
                corrupted = cor.mode != "none" and observed[i] == 0.0 and infos[i]["contact_force_norm"] > 0.5
                col = tuple(int(PALETTE[j * 7 % 8][q : q + 2], 16) for q in (1, 3, 5))
                final = outcome if i == len(frames) - 1 else None
                panels.append(annotate(frames[i], lab, col, infos[i], observed[i], final, limit, corrupted))
            video.append(np.concatenate(panels, axis=1))
        video += [video[-1]] * 20  # hold the last frame for one second
        tag = a.name or f"{a.domain}_ep{ep}"
        imageio.mimsave(out / f"{tag}.mp4", video, fps=20, macro_block_size=8)
        imageio.mimsave(out / f"{tag}.gif", [v[::2, ::2] for v in video[::2]], duration=0.1, loop=0)
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True)
        for ax, (_fr, infos, forces, observed, outcome, env), (_, lab, _) in zip(axes, runs, specs):
            t = np.arange(1, len(forces) + 1) * env.cfg.control_dt
            modes = [inf["contact_mode"] for inf in infos[1:]]
            for q in range(len(t)):
                ax.axvspan(t[q] - env.cfg.control_dt, t[q], color=MODE_COLORS[modes[q]], alpha=0.25, lw=0)
            ax.plot(t, forces, color="#0b0b0b", lw=1.6, label="true peak |F| per step")
            ax.plot(t, observed[1:], color=PALETTE[0], lw=1.0, label="observed |F| (policy input)")
            ax.axhline(limit, color="#d03b3b", ls=":", lw=1)
            ax.set_ylabel("force [N]")
            ax.set_title(f"{lab}: {outcome}", loc="left", fontsize=10)
        axes[0].legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("time [s]")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in MODE_COLORS]
        fig.legend(handles, MODE_NAMES, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(out / f"{tag}_force.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out / tag}.mp4/.gif/_force.png: " + ", ".join(f"{s[1]}={r[4]}" for r, s in zip(runs, specs)))


if __name__ == "__main__":
    main()
