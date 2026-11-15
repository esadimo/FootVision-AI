"""
Fine-tune a YOLO detector on SoccerNet-Tracking (player / goalkeeper / referee / ball).

Usage:
    python scripts/train_detector.py --model models/detection/yolo26s.pt --imgsz 1280 --epochs 10
Best weights are copied from the trainer save_dir to models/detection/<name>.pt.
Resume: python scripts/train_detector.py --resume runs/detect/runs/detect/<name>/weights/last.pt --batch 4
"""
import argparse, os, shutil, sys

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detection/yolo26s.pt")
    ap.add_argument("--data", default="data/yolo_soccernet/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="fv_yolo26s_1280")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--resume", default="", help="path to a last.pt checkpoint to resume from")
    a = ap.parse_args()

    if a.resume:
        model = YOLO(a.resume)
        model.train(resume=True, device=a.device, batch=a.batch, workers=3)
        _copy_best(model, a.name)
        return
    model = YOLO(a.model)
    model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
                workers=3, cos_lr=True, close_mosaic=3, patience=50,
                # broadcast footage: no vertical flips, mild colour jitter, small scale changes
                fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.3, translate=0.1,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, mosaic=1.0, mixup=0.0,
                project="runs/detect", name=a.name, exist_ok=True, plots=True)
    _copy_best(model, a.name)


def _copy_best(model, name):
    # Ultralytics may nest the project dir (runs/detect/runs/detect/<name>); use the trainer's real save_dir
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    cands = [os.path.join(str(save_dir), "weights", "best.pt")] if save_dir else []
    cands += [os.path.join("runs/detect/runs/detect", name, "weights", "best.pt"),
              os.path.join("runs/detect", name, "weights", "best.pt")]
    for best in cands:
        if os.path.exists(best):
            dst = os.path.join("models/detection", f"{name}.pt")
            shutil.copy(best, dst)
            print("BEST_WEIGHTS", best, "->", dst)
            return
    print("WARNING: best.pt not found in", cands)


if __name__ == "__main__":
    sys.exit(main())
