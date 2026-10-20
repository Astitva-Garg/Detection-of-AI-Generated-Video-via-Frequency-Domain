import math

import numpy as np

from src.eval.evaluate import metrics


def test_perfect():
    m = metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["acc"] == 1 and m["auroc"] == 1 and m["tpr_fake"] == 1 and m["tnr_real"] == 1


def test_constant_prob():
    assert metrics([0, 1, 0, 1], [0.5] * 4)["auroc"] == 0.5


def test_single_class_no_crash():
    m = metrics([1, 1, 1], [0.9, 0.8, 0.2])
    assert math.isnan(m["auroc"]) and abs(m["acc"] - 2 / 3) < 1e-9
