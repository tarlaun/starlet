import logging
from pathlib import Path

import numpy as np

from starlet._internal.histogram.io import (
    COMPRESSED_SUFFIX,
    load_numpy_array,
    load_prefix_histogram,
    resolve_histogram_path,
)

logger = logging.getLogger("bucket_mvt")


class HistogramLoader:
    def __init__(self, hist_path):
        self.hist_path = Path(hist_path)
        self.prefix = None

    def load(self):
        self.hist_path = resolve_histogram_path(self.hist_path)
        logger.info("Loading histogram from %s", self.hist_path)
        self.prefix = load_prefix_histogram(self.hist_path.parent)
        return self.prefix


COVERAGE_BASENAME = "global_cover"
COVERAGE_FILENAME = COVERAGE_BASENAME + COMPRESSED_SUFFIX


def coverage_prefix_path(hist_path) -> Path:
    """Path of the bbox coverage histogram next to a vertex histogram file
    (``global_cover.npy.gz``; a legacy uncompressed ``.npy`` is also read)."""
    return Path(hist_path).with_name(COVERAGE_FILENAME)


def load_coverage_prefix(hist_path) -> "np.ndarray | None":
    """Prefix sum of the bbox coverage histogram stored next to ``hist_path``,
    or ``None`` for datasets tiled before coverage histograms existed."""
    try:
        coverage = load_numpy_array(Path(hist_path).with_name(COVERAGE_BASENAME))
    except FileNotFoundError:
        return None
    return coverage.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
