import numpy as np
from noise_floor import floor, retention
def test_floor_identical_runs(): assert floor([[1, 0, 1, 1]] * 3)["min_pair"] == 1.0
def test_floor_detects_flips():
    f = floor([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]]); assert f["flip_items"] == 2 and f["min_pair"] < 1.0
def test_retention_matches_scorecard():
    import scorecard as SC; b = np.array([1, 1, 0, 1], bool); c = np.array([1, 0, 1, 1], bool); assert retention(b, c) == SC.retention(b, c)["ret"]
