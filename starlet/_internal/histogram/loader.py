import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("bucket_mvt")


class HistogramLoader:
    def __init__(self, hist_path):
        self.hist_path = Path(hist_path)
        self.prefix = None

    def load(self):
        logger.info("Loading histogram from %s", self.hist_path)
        arr = np.load(self.hist_path, allow_pickle=False)

        # If the provided file is already a prefix histogram (the integral image
        # ``global_prefix.npy`` written by the tiling stage), use it directly;
        # ``global.npy`` is the raw histogram and still needs the cumulative
        # sums computed here.
        if self.hist_path.stem.endswith("_prefix"):
            self.prefix = arr
        else:
            self.prefix = arr.cumsum(axis=0).cumsum(axis=1)

        return self.prefix


COVERAGE_FILENAME = "global_cover.npy"


def coverage_prefix_path(hist_path) -> Path:
    """Path of the bbox coverage histogram next to a vertex histogram file."""
    return Path(hist_path).with_name(COVERAGE_FILENAME)


def load_coverage_prefix(hist_path) -> "np.ndarray | None":
    """Prefix sum of the bbox coverage histogram stored next to ``hist_path``,
    or ``None`` for datasets tiled before coverage histograms existed."""
    path = coverage_prefix_path(hist_path)
    if not path.exists():
        return None
    coverage = np.load(path, allow_pickle=False)
    return coverage.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
