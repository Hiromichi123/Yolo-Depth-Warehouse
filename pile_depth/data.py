import json
import math
from pathlib import Path
import cv2
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset


def dataset_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = path.parent / root
    cfg["path"] = str(root.resolve())
    cfg["nc"] = len(cfg["names"])
    for split in ("train", "val", "test"):
        if split in cfg:
            cfg[split] = str(root / cfg[split])
    return cfg


def metadata(root, split):
    records = json.loads((Path(root) / "depth" / f"{split}.json").read_text(encoding="utf-8"))
    if not isinstance(records, dict):
        raise ValueError("Depth JSON must be an object keyed by root-relative image path")
    return records


def validate_dataset(path):
    cfg = dataset_config(path)
    root = Path(cfg["path"])
    piles = {}
    summary = {}
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        records = metadata(root, split)
        images = sorted(p for p in Path(cfg[split]).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        expected = {p.relative_to(root).as_posix() for p in images}
        if set(records) != expected or not images:
            raise ValueError(f"{split}: missing/extra depth labels or no images: {set(records) ^ expected}")
        for key, record in records.items():
            p = (root / key).resolve()
            if not p.is_relative_to(root.resolve()):
                raise ValueError(f"Path outside dataset: {key}")
            image = cv2.imread(str(p))
            if image is None:
                raise ValueError(f"Unreadable image: {p}")
            label = root / "labels" / p.relative_to(root / "images").with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing segmentation label: {label}")
            rows = label.read_text().strip().splitlines()
            if not rows:
                raise ValueError(f"Positive depth supervision requires a non-empty pile: {label}")
            for row in rows:
                values = [float(x) for x in row.split()]
                if len(values) < 7 or (len(values) - 1) % 2 or not all(math.isfinite(x) for x in values):
                    raise ValueError(f"Invalid polygon: {label}")
                if values[0] != int(values[0]) or not 0 <= values[0] < cfg["nc"] or any(x < 0 or x > 1 for x in values[1:]):
                    raise ValueError(f"Invalid class/coordinates: {label}")
            depth, front, total = (record.get(k) for k in ("average_depth", "front_count", "total_count"))
            if isinstance(depth, bool) or not isinstance(depth, (float, int)) or not math.isfinite(depth) or depth < 1:
                raise ValueError(f"Invalid average_depth: {key}")
            if type(front) is not int or front < 1 or type(total) is not int or total < front:
                raise ValueError(f"Invalid counts: {key}")
            if not math.isclose(depth, total / front, rel_tol=1e-3, abs_tol=1e-3):
                raise ValueError(f"Depth/count inconsistency: {key}")
            bbox = record.get("group_bbox")
            h, w = image.shape[:2]
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(type(x) in (int, float) and math.isfinite(x) for x in bbox):
                raise ValueError(f"Invalid group_bbox: {key}")
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"group_bbox outside image: {key}")
            pile = record.get("pile_id")
            if not isinstance(pile, str) or not pile:
                raise ValueError(f"Invalid pile_id: {key}")
            if pile in piles and piles[pile] != split:
                raise ValueError(f"pile_id leakage across splits: {pile}")
            piles[pile] = split
            if "chain_depths" in record:
                chains = record["chain_depths"]
                if len(chains) != front or any(type(x) is not int or x < 1 for x in chains) or sum(chains) != total:
                    raise ValueError(f"Invalid chain_depths: {key}")
        summary[split] = len(images)
    return summary


class DepthDataset(YOLODataset):
    def __init__(self, cfg, split, imgsz=640, batch_size=4):
        self.depth_root = Path(cfg["path"])
        self.records = metadata(self.depth_root, split)
        # Images MUST already be single-pile ROI images. group_bbox is annotation only.
        hyp = get_cfg(overrides={"task": "segment", "mosaic": 0., "mixup": 0., "copy_paste": 0.,
                                "degrees": 0., "translate": 0., "scale": 0., "shear": 0.,
                                "perspective": 0., "flipud": 0., "fliplr": 0.})
        super().__init__(img_path=cfg[split], imgsz=imgsz, batch_size=batch_size,
                         augment=False, hyp=hyp, rect=False, cache=False, data=cfg, task="segment")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        key = Path(item["im_file"]).relative_to(self.depth_root).as_posix()
        if key not in self.records:
            raise ValueError(f"Missing depth label: {key}")
        r = self.records[key]
        item["depth_target"] = torch.tensor([r["average_depth"]], dtype=torch.float32)
        item["front_count"] = torch.tensor([r["front_count"]], dtype=torch.float32)
        item["total_count"] = torch.tensor([r["total_count"]], dtype=torch.float32)
        return item

    @staticmethod
    def collate_fn(items):
        items = [dict(x) for x in items]
        extra = {key: torch.stack([x.pop(key) for x in items])
                 for key in ("depth_target", "front_count", "total_count")}
        return {**YOLODataset.collate_fn(items), **extra}


def prepare_batch(batch, device):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch["img"] = batch["img"].float() / 255.
    return batch
