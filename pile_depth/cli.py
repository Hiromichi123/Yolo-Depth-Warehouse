import argparse
import json
from pathlib import Path
import cv2
import torch
from .model import load_checkpoint, versions
from .data import dataset_config, validate_dataset
from .engine import train, depth_validate, native_validate
from .inference import infer, draw
from .exporter import export_onnx


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLO-seg + global geometric depth")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env")
    check = sub.add_parser("check-data")
    check.add_argument("--data", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--model", default="yolo11n-seg.pt")
    training.add_argument("--depth-epochs", type=int, default=10)
    training.add_argument("--joint-epochs", type=int, default=40)
    training.add_argument("--lambda-depth", type=float, default=.2)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", default="runs/train")
    val = sub.add_parser("val")
    val.add_argument("--data", required=True)
    val.add_argument("--checkpoint", required=True)
    val.add_argument("--split", choices=("val", "test"), default="val")
    val.add_argument("--output", default="runs/val.json")
    prediction = sub.add_parser("infer")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--front-count", type=int, help="Optional metadata; total uses visible count")
    prediction.add_argument("--roi", nargs=4, type=int, help="Optional search region; depth ROI is derived from detected cargo")
    prediction.add_argument("--conf", type=float, default=.25)
    prediction.add_argument("--iou", type=float, default=.7)
    prediction.add_argument("--classes", nargs="+", type=int, help="Cargo class IDs; omit for a cargo-only model")
    prediction.add_argument("--max-det", type=int, default=1000, help="Detection limit for dense piles")
    prediction.add_argument("--output", default="runs/prediction.json")
    exporting = sub.add_parser("export")
    exporting.add_argument("--checkpoint", required=True)
    exporting.add_argument("--output", default="runs/model.onnx")
    exporting.add_argument("--atol", type=float, default=1e-3)
    for p in (training, val, prediction, exporting):
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--device", default="cpu")
    for p in (training, val):
        p.add_argument("--batch", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "env":
        print(json.dumps({**versions(), "cuda_available": torch.cuda.is_available(),
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))
    elif args.command == "check-data":
        print(json.dumps(validate_dataset(args.data), indent=2))
    elif args.command == "train":
        if min(args.depth_epochs, args.joint_epochs) < 0:
            parser.error("Epoch counts cannot be negative")
        train(args)
    else:
        model, payload = load_checkpoint(args.checkpoint, args.device)
        if args.command == "export":
            print(json.dumps(export_onnx(model, args.output, args.imgsz, args.atol), indent=2))
        elif args.command == "val":
            validate_dataset(args.data)
            cfg = dataset_config(args.data)
            report = depth_validate(model, cfg, args.split, args.imgsz, args.batch,
                                    payload["training"].get("baseline_train_mean"))
            report["segmentation"] = native_validate(model, args.data, args.imgsz, args.batch, args.device, args.split)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))
        else:
            image = cv2.imread(args.image)
            if image is None:
                raise ValueError(f"Unreadable image: {args.image}")
            result = infer(model, image, args.front_count, args.roi, args.imgsz, args.conf,
                           args.iou, args.classes, args.max_det)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            if not cv2.imwrite(str(Path(args.output).with_suffix(".jpg")), draw(image, result)):
                raise OSError("Failed to save visualization")
            if result["group_bbox"] is not None:
                x1, y1, x2, y2 = result["group_bbox"]
                if not cv2.imwrite(str(Path(args.output).with_suffix(".roi.png")), image[y1:y2, x1:x2]):
                    raise OSError("Failed to save depth ROI")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
