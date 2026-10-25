#!/usr/bin/env python3
"""Open-loop evaluation of a g1_carry LoRA checkpoint (scripts/train/g1_carry_training.sh) on held-out episodes of a
WAM_CT LeRobot export, block by block as at deployment (one robot, KV cache kept): the first call sends frame 0 and the
state at row 0, call c > 0 the 9 frames of rows 8(c-1) .. 8c and the state (with its plan) at row 8c; each call predicts
the actions of rows 8c .. 8c+7. The model restarts its cache by itself every 4 blocks (as its training clips: a first
frame and 4 blocks).

The checkpoint holds only the trained weights (LoRA, state/action encoders and decoder), trained on top of
DreamZero-AgiBot: they are loaded onto AgiBot's weights (load_lora would load Wan's DiT under them instead).

Writes OUT/metrics.json (action errors: pelvis step xy cm, yaw deg, height cm, palm position cm, palm rotation deg;
per episode, overall and per step of the block), OUT/episodes/<episode>.npz (pred and gt, the predicted and true
actions, rows x 18) and OUT/videos/<episode>.mp4 (the real head camera left, the imagined one right) for the first
--videos episodes.

Usage (from the repo root, one GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_g1_carry.py --ckpt <run>/checkpoint-5000 --agibot <DreamZero-AgiBot> \
      --data <lerobot_carry2/val> --out <dir> [--episodes 16 | --pick 3 41 ...] [--videos 4]
"""

import torch._dynamo
torch._dynamo.config.disable = True

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from einops import rearrange
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation
from tianshou.data import Batch

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

BLOCK = 8  # rows (expert steps, 6.25 Hz) per block


def lora_on_agibot(agibot):
    """VLA.load_lora replacement: our config, AgiBot's weights, LoRA injected, then the checkpoint's trained weights."""
    def load(cls, path):
        cfg = json.loads((Path(path) / "config.json").read_text())
        head = cfg["action_head_cfg"]["config"]
        head["skip_component_loading"], head["defer_lora_injection"] = True, True
        model = cls(VLAConfig(**cfg))
        own = model.state_dict()
        index = json.loads((Path(agibot) / "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            # not AgiBot's weights of another shape (its state encoder, for another max_state_dim): the checkpoint has ours
            model.load_state_dict({k: v for k, v in load_file(Path(agibot) / shard).items() if k not in own or own[k].shape == v.shape},
                                  strict=False)
        model.action_head.inject_lora_after_loading()
        missing, unexpected = model.load_state_dict(load_file(Path(path) / "model.safetensors"), strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        return model
    return classmethod(load)


def quat_angle(a, b):
    """Angle (deg) between wxyz quaternions (..., 4)."""
    ra = Rotation.from_quat(a.reshape(-1, 4)[:, [1, 2, 3, 0]])
    rb = Rotation.from_quat(b.reshape(-1, 4)[:, [1, 2, 3, 0]])
    return np.degrees((ra.inv() * rb).magnitude()).reshape(a.shape[:-1])


def errors(pred, gt):
    """Per action row: pelvis step xy (cm), yaw (deg), height (cm), palm position (cm), palm rotation (deg; both palms
    averaged)."""
    return np.stack([
        100 * np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1),
        np.degrees(np.abs(np.angle(np.exp(1j * (pred[:, 2] - gt[:, 2]))))),
        100 * np.abs(pred[:, 3] - gt[:, 3]),
        100 * (np.linalg.norm(pred[:, 4:7] - gt[:, 4:7], axis=-1) + np.linalg.norm(pred[:, 11:14] - gt[:, 11:14], axis=-1)) / 2,
        (quat_angle(pred[:, 7:11], gt[:, 7:11]) + quat_angle(pred[:, 14:18], gt[:, 14:18])) / 2,
    ], axis=-1)


NAMES = ["pelvis_xy_cm", "yaw_deg", "height_cm", "palm_pos_cm", "palm_rot_deg"]


def decode(policy, latents):
    head = policy.trained_model.action_head
    frames = head.vae.decode(torch.cat(latents, dim=2), tiled=head.tiled, tile_size=(head.tile_size_height, head.tile_size_width),
                             tile_stride=(head.tile_stride_height, head.tile_stride_width))
    return ((rearrange(frames, "B C T H W -> B T H W C")[0].float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)


def run_episode(policy, df, frames, modality, plan_key, video):
    """Predicted and true actions (rows 0 .. 8C-1), and the side-by-side frames when video."""
    state = np.stack(df["observation.state"])
    plan = np.stack(df[plan_key]) if plan_key else None
    actions = np.stack(df["action"])
    text = str(df["annotation.task"].iloc[0])
    K = len(df)
    preds, side, window, start = [], [], [], 0
    for c in range((K - BLOCK) // BLOCK + 1):
        r = BLOCK * c
        obs = {"video.head": frames[[0]] if c == 0 else frames[r - BLOCK : r + 1], "annotation.task": text}
        for name, m in modality["state"].items():
            if name == "plan":
                if plan_key:
                    obs["state.plan"] = plan[r : r + 1]
            else:
                obs[f"state.{name}"] = state[r : r + 1, m["start"] : m["end"]]
        out, video_pred = policy.lazy_joint_forward_causal(Batch(obs=obs))
        preds.append(np.concatenate([out.act["action.pelvis"], out.act["action.palms"]], axis=-1).reshape(BLOCK, -1))
        if video:
            if video_pred.shape[2] > 2 and window:  # the cache restarted: decode the finished window
                side.append(decode(policy, window)[1 if side else 0 :])  # a later window starts at the last row of the previous
                window = []
            window.append(video_pred)
    if video and window:
        side.append(decode(policy, window)[1 if side else 0 :])
    pred = np.concatenate(preds)
    out = dict(pred=pred, gt=actions[: len(pred)])
    if video:
        imagined = np.concatenate(side)
        real = frames[: len(imagined)]
        out["video"] = np.concatenate([real, np.full((*real.shape[:2], 8, 3), 255, np.uint8), imagined[: len(real)]], axis=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--agibot", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--pick", type=int, nargs="+", help="these episodes instead of --episodes evenly spaced ones")
    ap.add_argument("--videos", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    VLA.load_lora = lora_on_agibot(args.agibot)
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.G1_CARRY, model_path=args.ckpt, device="cuda")

    data = Path(args.data)
    modality = json.loads((data / "meta/modality.json").read_text())
    state_keys = list(policy.train_cfg.modality_configs.g1_carry.state.modality_keys)
    plan_key = modality["state"]["plan"]["original_key"] if "state.plan" in state_keys else None
    sources = json.loads((data / "meta/sources.json").read_text())
    pick = args.pick or np.linspace(0, len(sources) - 1, min(args.episodes, len(sources))).round().astype(int)
    results, per_step = [], []
    for n, e in enumerate(pick):
        df = pd.read_parquet(data / f"data/chunk-000/episode_{e:06d}.parquet")
        # not list(reader): its length is inf, and list() would allocate for it
        frames = np.stack([f for f in iio.get_reader(data / f"videos/chunk-000/observation.images.head/episode_{e:06d}.mp4")])
        r = run_episode(policy, df, frames, modality, plan_key, video=n < args.videos)
        np.savez(out / "episodes" / f"episode_{e:06d}.npz", pred=r["pred"], gt=r["gt"])
        err = errors(r["pred"], r["gt"])  # (rows, 5)
        per_step.append(err.reshape(-1, BLOCK, len(NAMES)))
        results.append({"episode": int(e), "npz": sources[e]["npz"], "robot": sources[e]["robot"],
                        **{k: float(v) for k, v in zip(NAMES, err.mean(0))}})
        print(f"{n + 1}/{len(pick)} episode {e} ({sources[e]['npz']}, robot {sources[e]['robot']}): "
              + ", ".join(f"{k} {v:.2f}" for k, v in zip(NAMES, err.mean(0))), flush=True)
        if "video" in r:
            iio.mimsave(out / "videos" / f"episode_{e:06d}.mp4", list(r["video"]), fps=6.25, codec="libx264", macro_block_size=8)
    steps = np.concatenate(per_step)  # (blocks, 8, 5)
    summary = {"checkpoint": args.ckpt, "plan": plan_key, "episodes": results,
               "overall": {k: float(v) for k, v in zip(NAMES, steps.reshape(-1, len(NAMES)).mean(0))},
               "per_step_of_block": {k: steps[..., i].mean(0).round(3).tolist() for i, k in enumerate(NAMES)}}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print("overall: " + ", ".join(f"{k} {v:.2f}" for k, v in summary["overall"].items()))


if __name__ == "__main__":
    main()
