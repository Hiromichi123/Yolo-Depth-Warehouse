from pathlib import Path
import copy
import platform
import torch
from torch import nn
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Segment
from ultralytics.nn.tasks import SegmentationModel


def versions():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "ultralytics": ultralytics.__version__}


class DepthHead(nn.Sequential):
    def __init__(self, channels, hidden=128):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, feature):
        return 1 + torch.nn.functional.softplus(super().forward(feature))


class PileModel(nn.Module):
    """Observe the actual Segment input selected by stride, without fixed layer indices."""
    def __init__(self, segmenter, lambda_depth=0.2):
        super().__init__()
        self.segmenter = segmenter
        heads = [m for m in segmenter.modules() if isinstance(m, Segment)]
        if len(heads) != 1:
            raise ValueError("Exactly one native Segment head is required")
        head = heads[0]
        self.feature_slot = int(head.stride.argmax())
        channels = head.cv2[self.feature_slot][0].conv.in_channels
        self.depth_head = DepthHead(channels)
        self.channels = channels
        self.lambda_depth = lambda_depth
        self.frozen = False
        self._feature = None
        head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._feature = args[0][self.feature_slot]

    def forward(self, images):
        self._feature = None
        segmentation = self.segmenter(images)
        if self._feature is None:
            raise RuntimeError("Segment feature hook did not execute")
        depth = self.depth_head(self._feature)
        self._feature = None
        return segmentation, depth

    def segment_only(self, images):
        """Run localization without evaluating the depth head on the full scene."""
        self._feature = None
        try:
            return self.segmenter(images)
        finally:
            self._feature = None

    def set_stage(self, stage):
        if stage not in ("depth", "joint"):
            raise ValueError(stage)
        self.frozen = stage == "depth"
        for p in self.segmenter.parameters():
            p.requires_grad_(not self.frozen)
        # Native DFL projection is a fixed integral, even during joint fine-tuning.
        for name, p in self.segmenter.named_parameters():
            if ".dfl." in name:
                p.requires_grad_(False)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.frozen:
            # Keep native training return format but freeze all running statistics.
            for m in self.segmenter.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def losses(self, batch, predictions=None):
        segmentation, depth = self(batch["img"]) if predictions is None else predictions
        target = batch["depth_target"]
        if target.shape != depth.shape or not torch.isfinite(target).all() or (target < 1).any():
            raise ValueError("depth_target must be finite [B,1] with values >= 1")
        native_loss, items = self.segmenter.loss(batch, preds=segmentation)
        seg_loss = native_loss / len(depth)
        depth_loss = torch.nn.functional.smooth_l1_loss(depth, target)
        return seg_loss + self.lambda_depth * depth_loss, {
            "seg": seg_loss.detach(), "depth": depth_loss.detach(), "native": items.detach()}


def create_model(source="yolo11n-seg.yaml", nc=None, lambda_depth=0.2):
    segmenter = (SegmentationModel(copy.deepcopy(source), ch=3, verbose=False)
                 if isinstance(source, dict) else YOLO(str(source), task="segment").model).float()
    if nc is not None and int(segmenter.model[-1].nc) != nc:
        replacement = SegmentationModel(copy.deepcopy(segmenter.yaml), ch=3, nc=nc, verbose=False)
        replacement.load(segmenter)
        segmenter = replacement
    segmenter.args = get_cfg(overrides={"task": "segment"})
    return PileModel(segmenter, lambda_depth)


def save_checkpoint(model, path, training=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": 1, "versions": versions(), "yaml": model.segmenter.yaml,
                "names": model.segmenter.names,
                "head": {"channels": model.channels, "slot": model.feature_slot,
                         "hidden": 128, "activation": "1+softplus"},
                "lambda_depth": model.lambda_depth,
                "state_dict": model.state_dict(), "training": training or {}}, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    if payload["versions"]["ultralytics"] != ultralytics.__version__:
        raise ValueError("Ultralytics version mismatch; use the locked environment")
    model = create_model(payload["yaml"], lambda_depth=payload["lambda_depth"])
    expected = {"channels": model.channels, "slot": model.feature_slot,
                "hidden": 128, "activation": "1+softplus"}
    if payload["head"] != expected:
        raise ValueError("Depth head metadata mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.segmenter.names = payload["names"]
    return model.to(device).eval(), payload
