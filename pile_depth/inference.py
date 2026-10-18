import math
import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops


def summarize(instances, depth, front_count=None, group_bbox=None):
    if front_count is not None and (type(front_count) is not int or front_count < 1):
        raise ValueError("front_count must be a positive integer")
    visible = len(instances)
    average = float(depth) if visible else None
    if average is not None and (not math.isfinite(average) or average < 1):
        raise ValueError("average_depth must be finite and >= 1")
    total = math.floor(visible * average) if visible else None
    return {"visible_count": visible, "front_count": front_count,
            "average_depth": average, "total_count": total,
            "count_source": "visible_count_x_average_depth_floor" if total is not None else "unavailable",
            "group_bbox": group_bbox, "instances": instances}


def image_tensor(image, imgsz, device):
    if imgsz % 32:
        raise ValueError("imgsz must be divisible by 32")
    padded = LetterBox((imgsz, imgsz), auto=False)(image=image)
    return torch.from_numpy(np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))).to(device).float()[None] / 255.


def enclosing_bbox(instances, image_shape):
    """Union of detected boxes, rounded outward for an inclusive pixel crop."""
    if not instances:
        return None
    boxes = np.asarray([item["bbox"] for item in instances], dtype=float)
    h, w = image_shape[:2]
    x1 = max(0, math.floor(float(boxes[:, 0].min())))
    y1 = max(0, math.floor(float(boxes[:, 1].min())))
    x2 = min(w, math.ceil(float(boxes[:, 2].max())))
    y2 = min(h, math.ceil(float(boxes[:, 3].max())))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detected cargo union is empty")
    return [x1, y1, x2, y2]


@torch.no_grad()
def segment_instances(model, image, search_bbox, imgsz, conf, iou, classes, max_det):
    """Keep first-pass instances in original image coordinates."""
    x1, y1, x2, y2 = search_bbox
    roi = image[y1:y2, x1:x2]
    tensor = image_tensor(roi, imgsz, next(model.parameters()).device)
    raw = model.segment_only(tensor)
    nc = int(model.segmenter.model[-1].nc)
    if classes is not None and (not classes or any(type(c) is not int or not 0 <= c < nc for c in classes)):
        raise ValueError(f"classes must contain class IDs in [0, {nc - 1}]")
    detections = ops.non_max_suppression(raw[0], conf_thres=conf, iou_thres=iou,
                                        nc=nc, classes=classes, max_det=max_det)[0]
    proto = raw[1][-1][0]
    instances = []
    if len(detections):
        masks = ops.process_mask(proto, detections[:, 6:], detections[:, :4], tensor.shape[2:], upsample=True)
        masks = ops.scale_masks(masks[None].float(), roi.shape[:2])[0] > 0.5
        boxes = ops.scale_boxes(tensor.shape[2:], detections[:, :4], roi.shape[:2]).cpu().tolist()
        for index, (detection, box, mask) in enumerate(zip(detections, boxes, masks)):
            contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [(c.reshape(-1, 2) + [x1, y1]).tolist() for c in contours if len(c) >= 3]
            instances.append({"id": index + 1, "class_id": int(detection[5]),
                              "confidence": float(detection[4]),
                              "bbox": [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1],
                              "mask_polygons": polygons})
    return instances


@torch.no_grad()
def infer(model, image, front_count=None, group_bbox=None, imgsz=640, conf=0.25, iou=0.7,
          classes=None, max_det=1000):
    """Detect cargo -> crop enclosing box -> rerun depth on that crop only."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("image must be a nonempty BGR image")
    if front_count is not None and (type(front_count) is not int or front_count < 1):
        raise ValueError("front_count must be a positive integer")
    if type(max_det) is not int or max_det < 1:
        raise ValueError("max_det must be a positive integer")
    h, w = image.shape[:2]
    bbox = [0, 0, w, h] if group_bbox is None else group_bbox
    if len(bbox) != 4 or any(type(x) is not int for x in bbox):
        raise ValueError("ROI must contain four integer pixel coordinates")
    x1, y1, x2, y2 = bbox
    if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
        raise ValueError("ROI outside image")
    model.eval()
    instances = segment_instances(model, image, bbox, imgsz, conf, iou, classes, max_det)
    depth_bbox = enclosing_bbox(instances, image.shape)
    depth_value = None
    if depth_bbox is not None:
        dx1, dy1, dx2, dy2 = depth_bbox
        depth_input = image_tensor(image[dy1:dy2, dx1:dx2], imgsz, next(model.parameters()).device)
        _, depth = model(depth_input)
        depth_value = depth.item()
    result = summarize(instances, depth_value, front_count, depth_bbox)
    result.update({"search_bbox": list(bbox),
                   "depth_roi_source": "detected_cargo_union" if instances else "unavailable",
                   "cargo_classes": classes})
    return result


def draw(image, result):
    canvas = image.copy()
    overlay = image.copy()
    for item in result["instances"]:
        polygons = [np.array(p, dtype=np.int32) for p in item["mask_polygons"]]
        if polygons:
            cv2.fillPoly(overlay, polygons, (0, 180, 0))
    canvas = cv2.addWeighted(overlay, .35, canvas, .65, 0)
    h, w = canvas.shape[:2]

    def text(value, x, y, color):
        scale = min(.6, max(.1, (w - 6) / max(1, cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]) * .6))
        (tw, th), baseline = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(canvas, value, (max(0, min(int(x), w - tw - 1)),
                    max(th, min(int(y), h - baseline - 1))), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    for item in result["instances"]:
        x1, y1, x2, y2 = map(round, item["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text(str(item["id"]), x1, y1 - 4, (0, 255, 0))
    if result["group_bbox"]:
        x1, y1, x2, y2 = map(round, result["group_bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 2)
        text(f"total: {result['total_count'] if result['total_count'] is not None else 'N/A'}", x1, y2 - 8, (255, 0, 0))
    for idx, key in enumerate(("average_depth", "total_count", "visible_count")):
        value = result[key]
        shown = "N/A" if value is None else f"{value:.2f}" if key == "average_depth" else str(value)
        title = ("Avg Depth", "Total", "Visible")[idx]
        text(f"{title}: {shown}", 6, 22 + 23 * idx, (0, 255, 0))
    return canvas
