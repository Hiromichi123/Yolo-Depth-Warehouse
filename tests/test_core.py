import json
from pathlib import Path
import numpy as np
import pytest
import torch
from pile_depth.model import DepthHead, create_model, save_checkpoint, load_checkpoint
from pile_depth.data import validate_dataset, dataset_config, prepare_batch
from pile_depth.engine import loader, depth_validate
from pile_depth.inference import summarize, infer, draw


def test_head_shape_bounds_gradient():
    head = DepthHead(16)
    x = torch.randn(3, 16, 4, 4, requires_grad=True)
    y = head(x)
    assert y.shape == (3, 1) and (y >= 1).all()
    y.sum().backward()
    assert x.grad.abs().sum() > 0


def test_empty_and_unknown_front():
    empty = summarize([], 2.5, 4)
    assert empty["average_depth"] is None and empty["total_count"] is None
    result = summarize([{}] * 5, 3.0)
    assert result["total_count"] == 15 and result["front_count"] is None
    assert result["count_source"] == "visible_count_x_average_depth_floor"
    assert summarize([{}] * 5, 3., 2)["total_count"] == 15
    assert summarize([{}] * 3, 2.6)["total_count"] == 7
    with pytest.raises(ValueError):
        summarize([{}], 2., 0)


def test_data_and_alignment(dataset):
    assert validate_dataset(dataset) == {"train": 2, "val": 2, "test": 2}
    cfg = dataset_config(dataset)
    batch = next(iter(loader(cfg, "train", 64, 2)))
    assert batch["depth_target"].tolist() == [[2.], [3.]]
    assert batch["front_count"].tolist() == [[2.], [2.]]


@pytest.mark.parametrize("fault", ["missing", "formula", "leakage", "bbox", "polygon"])
def test_invalid_data_rejected(dataset, fault):
    path = dataset.parent / "depth/val.json"
    records = json.loads(path.read_text())
    r = records["images/val/0.png"]
    if fault == "missing":
        del records["images/val/0.png"]
    elif fault == "formula":
        r["average_depth"] = 10.
    elif fault == "leakage":
        r["pile_id"] = "train-0"
    elif fault == "bbox":
        r["group_bbox"][2] = 1000
    else:
        (dataset.parent / "labels/val/0.txt").write_text("0 .1 .2 .3 .4")
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError):
        validate_dataset(dataset)


def test_native_train_freeze_joint_validate_reload(dataset, tmp_path):
    cfg = dataset_config(dataset)
    model = create_model(nc=1)
    batch = prepare_batch(next(iter(loader(cfg, "train", 64, 2))), "cpu")
    model.set_stage("depth")
    bn = [m for m in model.segmenter.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    running = bn.running_mean.clone()
    old_seg = next(model.segmenter.parameters()).detach().clone()
    old_head = model.depth_head[2].weight.detach().clone()
    optimizer = torch.optim.Adam(model.depth_head.parameters(), lr=.01)
    loss, _ = model.losses(batch)
    loss.backward()
    optimizer.step()
    assert torch.equal(running, bn.running_mean)
    assert torch.equal(old_seg, next(model.segmenter.parameters()))
    assert not torch.equal(old_head, model.depth_head[2].weight)
    model.zero_grad(set_to_none=True)
    model.set_stage("joint")
    loss, _ = model.losses(batch)
    loss.backward()
    assert next(model.segmenter.parameters()).grad.abs().sum() > 0
    broken = dict(batch, depth_target=torch.ones(2))
    with pytest.raises(ValueError):
        model.losses(broken)
    model.eval()
    with torch.no_grad():
        raw, expected = model(batch["img"])
    assert raw[0].shape[0] == 2 and raw[1][-1].shape[1] == 32
    save_checkpoint(model, tmp_path / "model.pt")
    restored, _ = load_checkpoint(tmp_path / "model.pt")
    with torch.no_grad():
        actual_seg, actual = restored(batch["img"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_seg[0], raw[0], rtol=0, atol=0)
    assert depth_validate(restored, cfg, imgsz=64, batch=2)["n"] == 2
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    result = infer(restored, image, imgsz=64, conf=1.)
    assert result["average_depth"] is None
    assert draw(image, result).shape == image.shape


def test_checkpoint_rejects_metadata(tmp_path):
    model = create_model(nc=1)
    path = tmp_path / "broken.pt"
    save_checkpoint(model, path)
    state = torch.load(path, weights_only=True)
    state["head"]["channels"] = 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(path)


def test_onnx_all_outputs(tmp_path):
    from pile_depth.exporter import export_onnx
    report = export_onnx(create_model(nc=1).eval(), tmp_path / "model.onnx", imgsz=64)
    assert len(report["checks"]) == 2


@pytest.mark.skipif(not Path("yolo11n-seg.pt").is_file(), reason="Official weights not downloaded")
def test_pretrained_segment_preserved():
    import cv2
    import ultralytics
    from ultralytics import YOLO
    native = YOLO("yolo11n-seg.pt").model.eval()
    extended = create_model("yolo11n-seg.pt").eval()
    tensor = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        original = native(tensor)
        prediction, depth = extended(tensor)
    torch.testing.assert_close(original[0], prediction[0], rtol=0, atol=0)
    torch.testing.assert_close(original[1][-1], prediction[1][-1], rtol=0, atol=0)
    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets/bus.jpg"))
    result = infer(extended, image, imgsz=320)
    assert result["visible_count"] > 0
    assert result["total_count"] == int(np.floor(result["visible_count"] * result["average_depth"]))
    assert any(x["mask_polygons"] for x in result["instances"])
    from pile_depth.inference import enclosing_bbox, image_tensor
    assert result["group_bbox"] == enclosing_bbox(result["instances"], image.shape)
    x1, y1, x2, y2 = result["group_bbox"]
    with torch.no_grad():
        _, expected_depth = extended(image_tensor(image[y1:y2, x1:x2], 320, "cpu"))
    assert result["average_depth"] == expected_depth.item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device/runtime unavailable")
def test_cuda_inference():
    model = create_model(nc=1).cuda().eval()
    with torch.no_grad():
        raw, depth = model(torch.zeros(1, 3, 64, 64, device="cuda"))
    assert depth.is_cuda and raw[0].is_cuda
