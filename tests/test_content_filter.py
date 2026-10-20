import numpy as np

from src.data.content_filter import BLOCKED_CLASSES, is_allowed, max_blocked_area


class _NoBoxes:
    """Stand-in for a YOLO model that detects nothing."""

    class _R:
        names = {0: "person"}

        class boxes:
            import torch
            xyxy = torch.zeros((0, 4))
            cls = torch.zeros((0,))

    def predict(self, *a, **k):
        return [self._R()]


def test_threshold():
    assert is_allowed(0.04) and not is_allowed(0.06)


def test_blocked_classes():
    assert {"person", "dog", "bird", "car", "boat", "chair", "tv"} <= BLOCKED_CLASSES
    assert len(BLOCKED_CLASSES) == 80


def test_blank_frames():
    assert max_blocked_area(np.zeros((16, 64, 64, 3), np.uint8), _NoBoxes()) == (0.0, None)
