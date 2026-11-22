"""
Build a YOLO-format detection dataset from SoccerNet-Tracking ground truth.

Classes: 0 player, 1 goalkeeper, 2 referee, 3 ball   (see src/data/soccernet.DET_CLASSES)

Images are symlinked (not copied).  Whole *games* are held out for validation so
that the same kits / stadium never appear in both splits.

Usage:
    python scripts/prepare_yolo_dataset.py \
        --src data/SoccerNet/tracking/train --out data/yolo_soccernet \
        --stride 5 --val_games 3 --holdout_seq SNMOT-062
"""

import argparse
import configparser
import os
import random
import sys
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data.soccernet import DET_CLASSES, list_sequences  # noqa: E402


def game_id(seq) -> str:
    p = os.path.join(seq.path, "gameinfo.ini")
    cfg = configparser.ConfigParser()
    cfg.read(p)
    s = cfg["Sequence"] if "Sequence" in cfg else {}
    return str(s.get("gameid", s.get("gameID", seq.name)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/SoccerNet/tracking/train")
    ap.add_argument("--out", default="data/yolo_soccernet")
    ap.add_argument("--stride", type=int, default=5, help="use every Nth frame")
    ap.add_argument("--val_stride", type=int, default=15, help="frame stride for validation images")
    ap.add_argument("--val_games", type=int, default=1)
    ap.add_argument("--holdout_seq", nargs="*", default=["SNMOT-062"],
                    help="sequences whose whole game goes to val (never trained on)")
    ap.add_argument("--other_as", default="player", choices=["player", "skip"],
                    help="how to label GT role 'other'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seqs = [s for s in list_sequences(args.src) if s.has_gt]
    by_game = defaultdict(list)
    for s in seqs:
        by_game[game_id(s)].append(s)

    forced = {game_id(s) for s in seqs if s.name in args.holdout_seq}
    others = sorted(set(by_game) - forced)
    random.Random(args.seed).shuffle(others)
    val_games = forced | set(others[: max(0, args.val_games - len(forced))])

    counts = defaultdict(int)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    for g, gseqs in by_game.items():
        split = "val" if g in val_games else "train"
        for s in gseqs:
            W, H = s.image_size
            gt = s.gt
            for fi, img in enumerate(s.frame_paths, start=1):
                if (fi - 1) % (args.val_stride if split == "val" else args.stride):
                    continue
                name = f"{s.name}_{fi:06d}"
                dst = os.path.join(args.out, "images", split, name + ".jpg")
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(img), dst)
                rows = gt[gt["frame"] == fi]
                lines = []
                for r in rows:
                    role = r["role"]
                    if role == "other":
                        if args.other_as == "skip":
                            continue
                        role = "player"
                    c = DET_CLASSES[role]
                    x1, y1 = max(0.0, r["x1"]), max(0.0, r["y1"])
                    x2, y2 = min(W, r["x2"]), min(H, r["y2"])
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                 f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                    counts[(split, role)] += 1
                with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                counts[(split, "images")] += 1

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {v}: {k}\n" for k, v in DET_CLASSES.items()))

    print("val games:", sorted(val_games))
    for k in sorted(counts):
        print(f"  {k[0]:5s} {k[1]:11s} {counts[k]}")
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
