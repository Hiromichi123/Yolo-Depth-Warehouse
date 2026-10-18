import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ultralytics.nn.modules import Segment
from ultralytics.utils import ops


class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        native, depth = self.model(images)
        return native[0], native[1], depth


def export_onnx(model, path, imgsz=640, atol=1e-3):
    if imgsz % 32:
        raise ValueError("imgsz must be divisible by 32")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild from state to keep live model hooks and native head flags untouched.
    from .model import create_model
    cloned = create_model(copy.deepcopy(model.segmenter.yaml), lambda_depth=model.lambda_depth)
    cloned.load_state_dict(model.state_dict(), strict=True)
    cloned.eval()
    for head in cloned.modules():
        if isinstance(head, Segment):
            head.export, head.format, head.dynamic = True, "onnx", True
    wrapped = ExportModel(cloned).eval()
    sample = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        wrapped(sample)  # warm native anchor caches
    torch.onnx.export(wrapped, sample, str(path), opset_version=17, input_names=["images"],
                      output_names=["segmentation", "prototypes", "average_depth"],
                      dynamic_axes={name: {0: "batch"} for name in ("images", "segmentation", "prototypes", "average_depth")},
                      do_constant_folding=True, dynamo=False)
    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checks = []
    for b in (1, 2):
        torch.manual_seed(42 + b)
        sample = torch.rand(b, 3, imgsz, imgsz)
        with torch.no_grad():
            expected = [x.numpy() for x in wrapped(sample)]
        actual = session.run(None, {"images": sample.numpy()})
        errors = []
        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(result, reference, atol=atol, rtol=1e-4)
            errors.append(float(np.max(np.abs(reference - result))))
        # Low threshold exercises NMS and masks even with random test weights.
        nc = int(cloned.segmenter.model[-1].nc)
        detections = [ops.non_max_suppression(torch.from_numpy(outputs[0].copy()), conf_thres=0.001,
                                             nc=nc, max_det=20) for outputs in (expected, actual)]
        for i in range(b):
            left, right = detections[0][i], detections[1][i]
            if left.shape != right.shape:
                raise AssertionError("ONNX detection count/shape differs")
            if len(left):
                torch.testing.assert_close(left, right, atol=atol, rtol=1e-4)
                masks = [ops.process_mask(torch.from_numpy(outputs[1][i]), det[:, 6:],
                                         det[:, :4], (imgsz, imgsz), upsample=True)
                         for outputs, det in zip((expected, actual), (left, right))]
                if masks[0].shape != masks[1].shape or not torch.equal(masks[0], masks[1]):
                    raise AssertionError("ONNX binary mask mismatch")
        checks.append({"batch": b, "max_absolute_errors": errors,
                       "mask_counts": [len(d) for d in detections[0]]})
    report = {"outputs": ["segmentation", "prototypes", "average_depth"],
              "imgsz": imgsz, "dynamic_batch": True, "atol": atol, "checks": checks,
              "note": "Custom exporter; native Ultralytics exporter is not extended."}
    path.with_suffix(".validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
