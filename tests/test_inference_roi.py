from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from pile_depth.inference import infer, image_tensor


class SceneModel(torch.nn.Module):
    """Known detector: two cargo objects and one unrelated object on a 64x64 scene."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.segmenter = SimpleNamespace(model=[SimpleNamespace(nc=2)])
        self.depth_inputs = []

    def segment_only(self, images):
        predictions = torch.zeros(1, 38, 3)
        predictions[0, :4] = torch.tensor([[20., 40., 58.], [20., 40., 58.],
                                                   [16., 16., 8.], [16., 16., 8.]])
        predictions[0, 4:6] = torch.tensor([[.95, .9, 0.], [0., 0., .99]])
        predictions[0, 6:] = 1
        return predictions, ([], None, torch.ones(1, 32, 16, 16))

    def forward(self, images):
        self.depth_inputs.append(images.clone())
        return None, (1 + images.mean() * 4).reshape(1, 1)


def test_cargo_union_filters_classes_and_excludes_background():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[12:48, 12:48] = 100
    model = SceneModel()
    result = infer(model, image, classes=[0], imgsz=64, front_count=99)
    assert result["visible_count"] == 2
    assert result["group_bbox"] == [12, 12, 48, 48]
    assert len(model.depth_inputs) == 1
    torch.testing.assert_close(model.depth_inputs[0], image_tensor(image[12:48, 12:48], 64, "cpu"))
    expected = 1 + 100 / 255 * 4
    assert result["average_depth"] == pytest.approx(expected)
    assert result["total_count"] == math.floor(2 * expected)
    # Changing only background outside detected cargo must not change depth.
    changed = np.full_like(image, 255)
    changed[12:48, 12:48] = 100
    again = infer(model, changed, classes=[0], imgsz=64)
    assert again["average_depth"] == result["average_depth"]


def test_empty_skips_roi_and_depth():
    model = SceneModel()
    result = infer(model, np.zeros((64, 64, 3), dtype=np.uint8), conf=1., imgsz=64)
    assert model.depth_inputs == []
    assert result["group_bbox"] is None
    assert result["average_depth"] is None and result["total_count"] is None


def test_search_roi_maps_union_and_instances_to_original():
    model = SceneModel()
    result = infer(model, np.zeros((100, 100, 3), dtype=np.uint8), group_bbox=[10, 20, 74, 84], classes=[0], imgsz=64)
    assert result["search_bbox"] == [10, 20, 74, 84]
    assert result["group_bbox"] == [22, 32, 58, 68]
    assert result["instances"][0]["bbox"] == [22., 32., 38., 48.]


@pytest.mark.parametrize("classes", [[-1], [2], []])
def test_invalid_cargo_classes(classes):
    with pytest.raises(ValueError, match="classes"):
        infer(SceneModel(), np.zeros((64, 64, 3), dtype=np.uint8), classes=classes, imgsz=64)
