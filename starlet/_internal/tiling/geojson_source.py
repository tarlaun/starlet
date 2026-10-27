from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
import io
import json
import logging
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import ijson
import numpy as np
import pandas as pd
import pyarrow as pa

from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.datasource import (
    DataSource,
    SpatialSample,
    _GEOJSON_SUFFIXES,
    _attach_geoparquet_metadata,
    _combine_spatial_samples,
    _normalize_decimal_columns,
    _reservoir_add,
    _spatial_sample_from_state,
    _split_sample_cap,
    _source_files,
)
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader

logger = logging.getLogger(__name__)


def is_geojson_path(path: str) -> bool:
    return path.lower().endswith(_GEOJSON_SUFFIXES)


@dataclass(frozen=True)
class GeoJSONSplit:
    """Byte range to read from one GeoJSON source."""

    path: str
    offset: int
    length: int


class GeoJSONSource(DataSource):
    """
    Streams GeoJSON / GeoJSONL as Arrow Tables, converting geometry to WKB.

    - For standard FeatureCollection GeoJSON, reads byte partitions in parallel.
    - For GeoJSON Lines (one Feature per line), reads and batches by line.
    - Geometry dicts → shapely.shape → WKB bytes (binary Arrow column 'geometry').
    - Attaches minimal GeoParquet metadata (version, primary_column, encoding, crs hint).
    """

    def __init__(
        self,
        path: str,
        batch_rows: int = 1_000,
        src_crs: str = "EPSG:4326",
        target_crs: Optional[str] = None,
        keep_null_geoms: bool = False,
    ):
        self.path = str(path)
        self._files = _source_files(self.path, _GEOJSON_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoJSON files found in {self.path}")
        self.batch_rows = int(batch_rows)
        self.src_crs = src_crs
        self.target_crs = target_crs
        self.keep_null_geoms = keep_null_geoms

        if target_crs:
            logger.warning(
                "target_crs requested (%s) but GeoJSONSource preserves source CRS; "
                "projection is handled by downstream stages.",
                target_crs,
            )

        self._schema: Optional[pa.Schema] = None
        self._crs_hint: Optional[str] = _feature_collection_crs_hint_from_file(self._files[0])
        if self._crs_hint:
            logger.info("GeoJSON %s declares crs %s", self._files[0], self._crs_hint)

        logger.info(
            "GeoJSONSource opened %s with %d files (batch_rows=%d, src_crs=%s)",
            path, len(self._files), self.batch_rows, self.src_crs
        )

    # ---------------- schema ---------------- #
    def schema(self) -> pa.Schema:
        if self._schema is None:
            first = self._read_first_batch()
            if first is None or first.num_rows == 0:
                # Empty input file. Create a minimal schema with geometry column.
                base = pa.schema([("geometry", pa.binary())])
                self._schema = _attach_geoparquet_metadata(
                    base, self._crs_hint or self.src_crs
                )
            else:
                # Lock schema with GeoParquet metadata
                self._schema = _attach_geoparquet_metadata(
                    first.schema, self._crs_hint or self.src_crs
                )

        return self._schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    # ---------------- iterator ---------------- #
    def create_splits(self) -> List[GeoJSONSplit]:
        target_partition_size = 32 * 1024 * 1024
        splits: List[GeoJSONSplit] = []
        for file_path in self._files:
            file_size = file_path.stat().st_size
            num_splits = max(1, (file_size + target_partition_size - 1) // target_partition_size)
            splits.extend(
                GeoJSONSplit(path=str(file_path), offset=offset, length=length)
                for offset, length in _geojson_partition_ranges(file_size, int(num_splits))
            )
        return splits

    def iter_tables(self, split: Optional[GeoJSONSplit] = None) -> Iterable[pa.Table]:
        batch_index = 0
        crs_value = self._crs_hint or self.src_crs

        import geopandas as gpd

        for features in self._iter_feature_batches_for_split(split):
            if not features:
                continue

            gdf = gpd.GeoDataFrame.from_features(features, crs=crs_value)
            geometry_col = pa.array(gdf.geometry.to_wkb(), type=pa.binary())

            props_df = gdf.drop(columns="geometry")
            props_df = _normalize_decimal_columns(props_df)
            props_table = pa.Table.from_pandas(props_df, preserve_index=False)

            table = (
                pa.table([geometry_col], names=["geometry"])
                if props_table.num_columns == 0
                else props_table.append_column("geometry", geometry_col)
            )

            # Attach GeoParquet metadata with CRS
            schema_with_geo = _attach_geoparquet_metadata(table.schema, crs_value)
            table = table.replace_schema_metadata(schema_with_geo.metadata)

            if split is None and not self._schema:
                self._schema = schema_with_geo

            if self._schema is not None:
                table = self._coerce_to_schema(table, self._schema)
            table = table.combine_chunks()

            logger.debug(
                "GeoJSON batch %d (%d rows) -> %d columns (including 'geometry')",
                batch_index,
                table.num_rows,
                len(table.column_names),
            )
            batch_index += 1
            yield table

    def _iter_feature_batches_for_split(
        self,
        split: Optional[GeoJSONSplit],
    ) -> Iterable[List[Dict[str, Any]]]:
        if split is None:
            for source_split in self.create_splits():
                yield from self._iter_feature_batches_for_split(source_split)
            return

        reader = GeoJSONPartitionReader(split.path, split.offset, split.length, batch_size=self.batch_rows)
        for feature_batch in reader.batches():
            yield [json.loads(feature) for feature in feature_batch]

    # ---------------- internal helpers ---------------- #
    def _read_first_batch(self) -> Optional[pa.Table]:
        """Read the first batch of features to establish the schema."""
        first_path = self._files[0]
        file_size = first_path.stat().st_size
        batches = GeoJSONPartitionReader(first_path, 0, file_size, batch_size=max(1, self.batch_rows)).batches()
        try:
            first_batch = next(batches)
            features = [json.loads(feature_str) for feature_str in first_batch]
        except StopIteration:
            logger.info("GeoJSON read returned 0 rows when inferring schema")
            return None
        finally:
            batches.close()

        rows_props: List[Dict[str, Any]] = []
        geometries: List[Any] = []

        for feat in features:
            rows_props.append(feat.get("properties") or {})
            geometries.append(feat.get("geometry", None))

        props_df = pd.DataFrame.from_records(rows_props)
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)

        wkb_list = _geometries_to_wkb(geometries)
        geometry_col = pa.array(wkb_list, type=pa.binary())

        if props_table.num_columns == 0:
            return pa.table([geometry_col], names=["geometry"])

        return props_table.append_column("geometry", geometry_col)

    def _coerce_to_schema(self, t: pa.Table, schema: pa.Schema) -> pa.Table:
        if t.schema.equals(schema):
            return t

        out_cols = []
        out_fields = []
        promoted = False
        for fld in schema:
            name = fld.name
            if name in t.column_names:
                col = t[name]
                if not col.type.equals(fld.type):
                    if pa.types.is_null(fld.type) and not pa.types.is_null(col.type):
                        fld = pa.field(
                            fld.name,
                            col.type,
                            nullable=True,
                            metadata=fld.metadata,
                        )
                        promoted = True
                    else:
                        try:
                            col = col.cast(fld.type)
                        except Exception:
                            logger.warning(
                                "Type mismatch for column '%s': %s -> %s (kept original)",
                                name, col.type, fld.type
                            )
                            fld = pa.field(
                                fld.name,
                                col.type,
                                nullable=fld.nullable,
                                metadata=fld.metadata,
                            )
                out_cols.append(col)
                out_fields.append(fld)
            else:
                out_cols.append(pa.nulls(t.num_rows, type=fld.type))
                out_fields.append(fld)

        out_schema = pa.schema(out_fields, metadata=schema.metadata)
        if promoted:
            self._schema = out_schema
        return pa.table(out_cols, schema=out_schema)

    @classmethod
    def read_spatial_sample(
        cls,
        path: str,
        *,
        sample_ratio: float,
        sample_cap: Optional[int],
        seed: int,
        workers: Optional[int],
    ) -> SpatialSample:
        source = cls(path)
        splits = source.create_splits()
        sample_caps = _split_sample_cap(sample_cap, len(splits))

        logger.info(
            "Reading GeoJSON spatial sample from %s in %d partitions with %s process workers",
            path,
            len(splits),
            workers or "auto",
        )

        with create_process_executor(
            max_workers=workers,
            logger=logger,
            context="GeoJSON spatial sampling",
        ) as ex:
            futures = [
                ex.submit(
                    _read_geojson_partition_spatial_sample,
                    split.path,
                    split.offset,
                    split.length,
                    sample_ratio,
                    sample_caps[idx],
                    seed + idx,
                )
                for idx, split in enumerate(splits)
            ]
            parts: List[SpatialSample] = []
            for future in as_completed(futures):
                parts.append(future.result())
            return _combine_spatial_samples(parts)


def _read_geojson_spatial_sample(
    path: str,
    *,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
    geojson_workers: Optional[int],
) -> SpatialSample:
    return GeoJSONSource.read_spatial_sample(
        path,
        sample_ratio=sample_ratio,
        sample_cap=sample_cap,
        seed=seed,
        workers=geojson_workers,
    )


def iter_geojson_xy(feature_json):
    try:
        geometry = next(
            ijson.items(
                io.BytesIO(feature_json.encode("utf-8")),
                "geometry",
                use_float=True,
            ),
            None,
        )
    except Exception:
        print("Failed to parse feature_json:")
        raise

    stack = [geometry]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if v.get("type") == "GeometryCollection":
                stack.extend(reversed(v.get("geometries") or []))
            else:
                coordinates = v.get("coordinates")
                if coordinates is not None:
                    stack.append(coordinates)
        elif isinstance(v, list):
            if len(v) >= 2 and isinstance(v[0], Number) and isinstance(v[1], Number):
                yield float(v[0]), float(v[1])
            else:
                stack.extend(reversed(v))


def _read_geojson_partition_spatial_sample(
    path: str,
    offset: int,
    length: int,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
    reader = GeoJSONPartitionReader(path, offset, length, batch_size=1_024)
    rng = np.random.default_rng(seed)
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    n_batches = 0

    for batch in reader:
        for feature_json in batch:
            first_point = True
            for x, y in iter_geojson_xy(feature_json):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

                if first_point:
                    first_point = False
                    n_seen += 1
                    _reservoir_add(
                        rng=rng,
                        sample_cap=sample_cap,
                        sample_ratio=sample_ratio,
                        x_sample=x_sample,
                        y_sample=y_sample,
                        n_seen=n_seen,
                        x=x,
                        y=y,
                    )

        n_batches += 1

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        mins=np.array([min_x, min_y], dtype=float),
        maxs=np.array([max_x, max_y], dtype=float),
        n_seen=n_seen,
        batches_read=n_batches,
    )


def _geometries_to_wkb(geometries: List[Any]) -> List[Any]:
    """
    Vectorized geometry -> WKB conversion using shapely's GeoJSON reader.

    Converting via shapely.geometry.shape per-feature is expensive for large
    files. Using shapely.from_geojson on an array of compact JSON strings keeps
    the heavy work inside GEOS and removes most Python-level loops.
    """
    from shapely import from_geojson, to_wkb

    wkb: List[Any] = [None] * len(geometries)
    non_null_idx: List[int] = []
    geojson_strings: List[str] = []

    for idx, geom in enumerate(geometries):
        if geom is None:
            continue
        non_null_idx.append(idx)
        geojson_strings.append(json.dumps(geom, separators=(",", ":")))

    if not geojson_strings:
        return wkb

    shapely_geoms = from_geojson(geojson_strings)
    encoded = to_wkb(shapely_geoms, hex=False).tolist()

    for idx, val in zip(non_null_idx, encoded):
        wkb[idx] = val

    return wkb


def _geojson_partition_ranges(file_size: int, num_splits: int) -> List[tuple[int, int]]:
    if file_size <= 0:
        return []

    num_splits = max(1, min(int(num_splits), file_size))
    partition_size = max(1, (file_size + num_splits - 1) // num_splits)
    ranges: List[tuple[int, int]] = []

    for offset in range(0, file_size, partition_size):
        ranges.append((offset, min(partition_size, file_size - offset)))

    return ranges


_CRS_SCAN_BYTES = 64 * 1024


def _feature_collection_crs_hint_from_file(path) -> Optional[str]:
    """CRS declared by a FeatureCollection's (deprecated but common) ``crs``
    member. Only the first and last 64 KiB of the file are read: the member
    sits either before ``"features"`` or after it."""
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
        with open(file_path, "rb") as handle:
            head = handle.read(_CRS_SCAN_BYTES)
            if size > _CRS_SCAN_BYTES:
                handle.seek(max(0, size - _CRS_SCAN_BYTES))
                tail = handle.read(_CRS_SCAN_BYTES)
            else:
                tail = b""
    except OSError:
        return None
    for chunk in (head, tail):
        name = _extract_feature_collection_crs_hint(chunk.decode("utf-8", errors="replace"))
        if name:
            return _validated_crs_name(name)
    return None


def _validated_crs_name(name: str) -> Optional[str]:
    """Return the CRS string if pyproj understands it, else ``None`` (with a
    warning) so a bad ``crs`` member never breaks reading."""
    from pyproj import CRS

    try:
        crs = CRS.from_user_input(name)
    except Exception:
        logger.warning("Ignoring unrecognised GeoJSON crs %r; assuming EPSG:4326", name)
        return None
    if crs.equals(CRS.from_epsg(4326), ignore_axis_order=True):
        return None  # lon/lat is the default; no reprojection needed
    return name


def _extract_feature_collection_crs_hint(buffer: str) -> Optional[str]:
    """Find a GeoJSON ``"crs": {...}`` member in ``buffer`` (any position) and
    return its ``properties.name``; ``None`` when absent or unparsable."""
    if not buffer:
        return None

    lowered = buffer.lower()
    start = 0
    while True:
        idx = lowered.find('"crs"', start)
        if idx == -1:
            return None
        colon = buffer.find(":", idx + 5)
        if colon == -1:
            return None
        brace = buffer.find("{", colon)
        if brace == -1 or buffer[colon + 1:brace].strip():
            start = idx + 5
            continue
        end = _matching_brace(buffer, brace)
        if end == -1:
            return None
        try:
            crs = json.loads(buffer[brace:end + 1])
        except Exception:
            start = idx + 5
            continue
        if isinstance(crs, dict):
            props = crs.get("properties") or {}
            name = props.get("name")
            if isinstance(name, str):
                return name
        start = end


def _matching_brace(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1
