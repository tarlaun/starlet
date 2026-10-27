"""Transparent handling of compressed single-file inputs (``.geojson.gz`` ...)."""
from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from starlet._internal.config import resolve_temp_dir

logger = logging.getLogger(__name__)

_OPENERS: dict[str, Callable] = {
    ".gz": gzip.open,
    ".bz2": bz2.open,
    ".xz": lzma.open,
}


# ``.bz2`` files the readers stream directly (block-parallel), so they are not
# decompressed up front.
_NATIVE_BZ2_INNER_SUFFIXES = (".geojson", ".geojsonl", ".json", ".jsonl", ".csv", ".tsv", ".txt")


def compressed_suffix(path: str | Path) -> str | None:
    """The compression suffix of ``path`` when starlet has to decompress it
    before reading, else ``None``."""
    name = Path(path).name.lower()
    suffix = Path(name).suffix
    if suffix not in _OPENERS:
        return None
    if suffix == ".bz2" and name[: -len(suffix)].endswith(_NATIVE_BZ2_INNER_SUFFIXES):
        return None
    return suffix


def is_compressed_path(path: str | Path) -> bool:
    return Path(path).is_file() and compressed_suffix(path) is not None


def decompress_to(path: str | Path, target_dir: str | Path) -> Path:
    """Decompress ``path`` into ``target_dir`` keeping the inner file name
    (``states.geojson.gz`` -> ``states.geojson``)."""
    source = Path(path)
    suffix = compressed_suffix(source)
    if suffix is None:
        raise ValueError(f"Not a compressed file: {path}")
    inner_name = source.name[: -len(suffix)] or source.stem
    target = Path(target_dir) / inner_name
    logger.info("Decompressing %s to %s", source, target)
    with _OPENERS[suffix](source, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    return target


@contextmanager
def materialized_input(path: str, temp_dir: str | Path | None = None) -> Iterator[str]:
    """Yield a path starlet's readers can open: ``path`` itself, or a
    temporary decompressed copy for ``.gz`` / ``.bz2`` / ``.xz`` inputs."""
    if not is_compressed_path(path):
        yield path
        return
    temp_parent = resolve_temp_dir(temp_dir, Path.cwd() / "tmp")
    work_dir = Path(tempfile.mkdtemp(prefix="starlet_decompress_", dir=str(temp_parent)))
    try:
        yield str(decompress_to(path, work_dir))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
