import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from .data import DepthDataset, dataset_config, prepare_batch, validate_dataset, metadata
from .model import create_model, load_checkpoint, save_checkpoint


def loader(cfg, split, imgsz, batch, shuffle=False):
    ds = DepthDataset(cfg, split, imgsz, batch)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, collate_fn=ds.collate_fn)


def measures(pred, truth, fronts, totals):
    pred, truth, fronts, totals = map(np.asarray, (pred, truth, fronts, totals))
    counts = np.rint(pred * fronts)
    return {"n": len(pred), "depth_mae": float(np.abs(pred - truth).mean()),
            "depth_rmse": float(np.sqrt(((pred - truth) ** 2).mean())),
            "total_mae": float(np.abs(counts - totals).mean()),
            "total_mape": float((np.abs(counts - totals) / totals).mean())}


@torch.no_grad()
def depth_validate(model, cfg, split="val", imgsz=640, batch=4, baseline=None):
    model.eval()
    predictions, targets, fronts, totals, files = [], [], [], [], []
    for data in loader(cfg, split, imgsz, batch):
        data = prepare_batch(data, next(model.parameters()).device)
        _, depth = model(data["img"])
        predictions.extend(depth[:, 0].cpu().tolist())
        targets.extend(data["depth_target"][:, 0].cpu().tolist())
        fronts.extend(data["front_count"][:, 0].cpu().tolist())
        totals.extend(data["total_count"][:, 0].cpu().tolist())
        files.extend(data["im_file"])
    report = measures(predictions, targets, fronts, totals)
    if baseline is not None:
        report["fixed_train_mean_baseline"] = measures([baseline] * len(targets), targets, fronts, totals)
    records = metadata(cfg["path"], split)
    groups = defaultdict(list)
    for i, f in enumerate(files):
        record = records[Path(f).relative_to(cfg["path"]).as_posix()]
        for field in ("camera_id", "package_type", "view_direction"):
            groups[f"{field}={record.get(field, 'unknown')}"].append(i)
        groups[f"depth_bin={math.floor(targets[i])}"].append(i)
        x1, y1, x2, y2 = record["group_bbox"]
        area = (x2 - x1) * (y2 - y1)
        groups[f"roi_area={'small' if area < 640**2 else 'large'}"].append(i)
    report["groups"] = {k: measures(*[[arr[i] for i in ids] for arr in (predictions, targets, fronts, totals)]) for k, ids in groups.items()}
    return report


def train(args):
    validate_dataset(args.data)
    cfg = dataset_config(args.data)
    torch.manual_seed(args.seed)
    model = create_model(args.model, cfg["nc"], args.lambda_depth).to(args.device)
    model.segmenter.names = dict(enumerate(cfg["names"])) if isinstance(cfg["names"], list) else cfg["names"]
    baseline = float(np.mean([r["average_depth"] for r in metadata(cfg["path"], "train").values()]))
    train_loader = loader(cfg, "train", args.imgsz, args.batch, True)
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for stage, epochs in (("depth", args.depth_epochs), ("joint", args.joint_epochs)):
        model.set_stage(stage)
        groups = [{"params": list(model.depth_head.parameters()), "lr": args.lr}]
        if stage == "joint":
            groups.append({"params": [p for p in model.segmenter.parameters() if p.requires_grad], "lr": args.lr * .1})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        for epoch in range(epochs):
            model.train()
            loss_sums = defaultdict(float)
            for batch in train_loader:
                batch = prepare_batch(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.losses(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                for name, module in (("depth", model.depth_head), ("seg", model.segmenter)):
                    loss_sums[f"grad_{name}"] += math.sqrt(sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                for k in ("seg", "depth"):
                    loss_sums[k] += float(parts[k])
            report = depth_validate(model, cfg, imgsz=args.imgsz, batch=args.batch, baseline=baseline)
            entry = {"stage": stage, "epoch": epoch + 1,
                     "train": {k: v / len(train_loader) for k, v in loss_sums.items()}, "val": report}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            info = {"baseline_train_mean": baseline, "history": history, "source": args.model,
                    "seed": args.seed, "data": str(Path(args.data).resolve())}
            save_checkpoint(model, run / "last.pt", info)
            if report["depth_mae"] < best:
                best = report["depth_mae"]
                save_checkpoint(model, run / "best.pt", info)
            (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history:
        raise ValueError("At least one training epoch required")
    # Native validator remains intact, including box/mask AP; validate best checkpoint.
    best_model, _ = load_checkpoint(run / "best.pt", args.device)
    report = native_validate(best_model, args.data, args.imgsz, args.batch, args.device, "val", run)
    (run / "segmentation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def native_validate(model, data, imgsz, batch, device, split="val", output="runs"):
    native = YOLO("yolo11n-seg.yaml", task="segment")
    native.model = model.segmenter
    native.overrides["model"] = "pile-depth-checkpoint.pt"
    result = native.val(data=str(Path(data).resolve()), imgsz=imgsz, batch=batch, device=device,
                        split=split, plots=False, workers=0, project=str(output), name="seg_val", exist_ok=True)
    return {k: float(v) for k, v in result.results_dict.items()}
