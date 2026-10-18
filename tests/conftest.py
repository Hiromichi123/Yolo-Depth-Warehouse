import json
import os
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import yaml

torch.set_num_threads(1)


def make_dataset(root):
    root = Path(root)
    for split in ("train", "val", "test"):
        records = {}
        for i in range(2):
            key = f"images/{split}/{i}.png"
            image = np.zeros((96, 128, 3), np.uint8)
            cv2.rectangle(image, (25, 20), (95, 75), (100, 180, 210), -1)
            (root / key).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / key), image)
            label = root / f"labels/{split}/{i}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.2 0.2 0.75 0.2 0.75 0.8 0.2 0.8\n")
            records[key] = {"pile_id": f"{split}-{i}", "average_depth": float(i + 2),
                            "front_count": 2, "total_count": 2 * (i + 2),
                            "group_bbox": [0, 0, 128, 96], "camera_id": "synthetic"}
        (root / "depth").mkdir(exist_ok=True)
        (root / f"depth/{split}.json").write_text(json.dumps(records), encoding="utf-8")
    config = root / "dataset.yaml"
    config.write_text(yaml.safe_dump({"path": str(root.resolve()), "train": "images/train",
                                     "val": "images/val", "test": "images/test", "names": ["package"]}), encoding="utf-8")
    return config


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path)
