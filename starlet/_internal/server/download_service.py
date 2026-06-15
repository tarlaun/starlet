"""
Download API Service for streaming geospatial features in CSV and GeoJSON formats.
Supports spatial filtering using Minimum Bounding Rectangle (MBR).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
import json
import pyarrow.parquet as pq
import pyarrow.compute as pc
from shapely import from_wkb
from shapely.geometry import box as _shp_box, mapping as _shp_mapping, shape as _shp_shape

from .tiler.parquet_index import parse_parquet_bbox as _parse_filename_bbox

# Per-row bbox covering columns written by the tiling stage (see writer_pool).
_BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


@dataclass
class BoundingBox:
    """Represents a spatial bounding box with intersection logic."""
    minx: float
    miny: float
    maxx: float
    maxy: float
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects with another."""
        return not (
            self.maxx < other.minx or 
            self.minx > other.maxx or 
            self.maxy < other.miny or 
            self.miny > other.maxy
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this bounding box."""
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy
    
    @classmethod
    def from_string(cls, mbr_string: str) -> "BoundingBox":
        """Parse MBR from string format: 'x1,y1,x2,y2'."""
        parts = [float(p) for p in mbr_string.split(',')]
        if len(parts) != 4:
            raise ValueError("MBR must be 4 values: minx,miny,maxx,maxy")
        return cls(minx=parts[0], miny=parts[1], maxx=parts[2], maxy=parts[3])


class TileManager:
    """Manages parquet tile discovery and filtering by MBR."""
    
    def __init__(self, dataset_path: Path):
        """Initialize with path to parquet_tiles directory."""
        self.dataset_path = dataset_path
        self.tiles_dir = dataset_path / "parquet_tiles"
        
    def parse_tile_mbr(self, filename: str) -> Optional[BoundingBox]:
        """
        Parse MBR from a tile filename.
        Format: tile_XXXXXX__minx_miny_maxx_maxy.parquet, where each coordinate
        is an ``int_decimal`` pair (e.g. ``-78_099403`` -> ``-78.099403``).
        """
        bb = _parse_filename_bbox(filename)
        if bb is None:
            return None
        return BoundingBox(minx=bb[0], miny=bb[1], maxx=bb[2], maxy=bb[3])
    
    def find_intersecting_tiles(self, query_mbr: Optional[BoundingBox]) -> List[Path]:
        """Find all parquet tiles that intersect with the query MBR.
        If query_mbr is None, return all tiles."""
        intersecting_tiles = []
        
        if not self.tiles_dir.exists():
            return intersecting_tiles
        
        # If no MBR specified, return all tiles
        if query_mbr is None:
            return sorted(self.tiles_dir.glob("*.parquet"))
        
        # Filter tiles by MBR intersection
        for tile_file in self.tiles_dir.glob("*.parquet"):
            tile_mbr = self.parse_tile_mbr(tile_file.name)
            if tile_mbr and query_mbr.intersects(tile_mbr):
                intersecting_tiles.append(tile_file)
        
        return sorted(intersecting_tiles)


class FormatHandler(ABC):
    """Abstract base class for feature format handlers."""
    
    def __init__(self, output_mbr: Optional[BoundingBox] = None):
        """Initialize handler with optional output MBR for filtering."""
        self.output_mbr = output_mbr
    
    @abstractmethod
    def initialize(self) -> str:
        """Return initialization content (headers, opening tags, etc.)."""
        pass
    
    @abstractmethod
    def write_feature(self, feature: Dict[str, Any]) -> str:
        """Convert a feature to output format string."""
        pass
    
    @abstractmethod
    def finalize(self) -> str:
        """Return finalization content (closing tags, etc.)."""
        pass
    
    def should_include_feature(self, feature: Dict[str, Any]) -> bool:
        """Filter feature by output MBR if specified."""
        if self.output_mbr is None:
            return True
        
        # Check if geometry exists and is a point
        if 'geometry' not in feature:
            return False
        
        geom = feature['geometry']
        if geom is None:
            return False
        
        # Handle Point geometry
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if len(coords) >= 2:
                x, y = coords[0], coords[1]
                return self.output_mbr.contains_point(x, y)
        
        return True


class CSVHandler(FormatHandler):
    """Handles CSV format output."""
    
    def __init__(self, output_mbr: Optional[BoundingBox] = None):
        super().__init__(output_mbr)
        self.fieldnames = None
        self.writer_initialized = False
    
    def initialize(self) -> str:
        return ""  # Headers written with first row
    
    def write_feature(self, feature: Dict[str, Any]) -> str:
        """Convert feature to CSV row."""
        if not self.should_include_feature(feature):
            return ""
        
        props = feature.get('properties', {})
        
        # Initialize fieldnames from first feature
        if self.fieldnames is None:
            self.fieldnames = list(props.keys())
            self.fieldnames.extend(['geometry_type', 'x', 'y'])
            # Write header
            header = ",".join(self.fieldnames) + "\n"
            return header + self._feature_to_csv(props, feature.get('geometry'))
        
        return self._feature_to_csv(props, feature.get('geometry'))
    
    def _feature_to_csv(self, properties: Dict, geometry: Optional[Dict]) -> str:
        """Convert properties dict to CSV row."""
        row = []
        for field in self.fieldnames:
            if field == 'geometry_type' and geometry:
                row.append(geometry.get('type', ''))
            elif field == 'x' and geometry:
                c = _first_coord(geometry.get('coordinates'))
                row.append(str(c[0]) if c else '')
            elif field == 'y' and geometry:
                c = _first_coord(geometry.get('coordinates'))
                row.append(str(c[1]) if c and len(c) > 1 else '')
            else:
                value = properties.get(field, '')
                # Escape quotes and wrap if needed
                if isinstance(value, str) and (',' in value or '"' in value):
                    value = f'"{value.replace(chr(34), chr(34)+chr(34))}"'
                row.append(str(value))
        
        return ",".join(row) + "\n"
    
    def finalize(self) -> str:
        return ""


class GeoJSONHandler(FormatHandler):
    """Handles GeoJSON format output."""
    
    def __init__(self, output_mbr: Optional[BoundingBox] = None):
        super().__init__(output_mbr)
        self.first_feature = True
    
    def initialize(self) -> str:
        return '{"type":"FeatureCollection","features":['
    
    def write_feature(self, feature: Dict[str, Any]) -> str:
        """Convert feature to GeoJSON format."""
        if not self.should_include_feature(feature):
            return ""
        
        if not self.first_feature:
            return "," + json.dumps(feature)
        else:
            self.first_feature = False
            return json.dumps(feature)
    
    def finalize(self) -> str:
        return "]}"


def _first_coord(coords):
    """Return the first ``[x, y]`` pair from (possibly nested) GeoJSON coords."""
    if not coords:
        return None
    if isinstance(coords[0], (int, float)):
        return coords
    return _first_coord(coords[0])


class FeatureStreamer:
    """Streams features from parquet tiles with spatial filtering.

    Spatial filtering happens in two stages, mirroring the tile server:
      1. partition pruning by filename bbox (``find_intersecting_tiles``);
      2. row-group + row pruning via pyarrow predicate pushdown on the
         ``_bbox_*`` covering columns (when present), then an exact
         geometry-intersection test against the query shape.
    WKB geometries are decoded so features carry real geometry (the previous
    implementation left geometry ``None``, dropping every filtered feature).
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.tile_manager = TileManager(dataset_path)

    def stream_features(
        self,
        query_mbr: Optional[BoundingBox],
        format_handler: FormatHandler,
        filter_geom=None,
    ) -> Iterator[str]:
        """Stream features intersecting ``filter_geom`` (or ``query_mbr``) as
        the requested format. With no filter, streams all features."""
        yield format_handler.initialize()

        tiles = self.tile_manager.find_intersecting_tiles(query_mbr)
        if not tiles:
            yield format_handler.finalize()
            return

        if filter_geom is None and query_mbr is not None:
            filter_geom = _shp_box(query_mbr.minx, query_mbr.miny, query_mbr.maxx, query_mbr.maxy)

        for tile_path in tiles:
            try:
                pf = pq.ParquetFile(str(tile_path))
                names = pf.schema_arrow.names
                if query_mbr is not None and all(c in names for c in _BBOX_COLS):
                    flt = ((pc.field("_bbox_xmax") >= query_mbr.minx)
                           & (pc.field("_bbox_xmin") <= query_mbr.maxx)
                           & (pc.field("_bbox_ymax") >= query_mbr.miny)
                           & (pc.field("_bbox_ymin") <= query_mbr.maxy))
                    table = pq.read_table(str(tile_path), filters=flt)
                else:
                    table = pq.read_table(str(tile_path))

                df = table.to_pandas()
                if "geometry" not in df.columns or len(df) == 0:
                    continue
                geoms = from_wkb(df["geometry"].to_numpy())
                prop_cols = [c for c in df.columns if c != "geometry" and not c.startswith("_bbox_")]
                records = df[prop_cols].to_dict("records")

                for geom, props in zip(geoms, records):
                    if geom is None or geom.is_empty:
                        continue
                    if filter_geom is not None and not filter_geom.intersects(geom):
                        continue
                    feature = {
                        "type": "Feature",
                        "properties": {k: v for k, v in props.items() if v is not None},
                        "geometry": _shp_mapping(geom),
                    }
                    output = format_handler.write_feature(feature)
                    if output:
                        yield output
            except Exception as e:
                print(f"Error reading tile {tile_path}: {e}")
                continue

        yield format_handler.finalize()


class DatasetFeatureService:
    """High-level service for dataset feature downloads."""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
    
    def get_features_stream(
        self,
        dataset_name: str,
        format: str,
        mbr_string: Optional[str] = None,
        geometry: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Get streaming response for features.

        Args:
            dataset_name: Name of dataset (e.g., 'TIGER2018_COUNTY')
            format: Output format ('csv' or 'geojson')
            mbr_string: Optional bounding box string 'minx,miny,maxx,maxy'.
            geometry: Optional GeoJSON geometry to filter by (exact intersection).
                      Takes precedence over mbr_string. If both are None, returns
                      all features.

        Yields:
            String chunks for streaming response
        """
        query_mbr = None
        filter_geom = None
        if geometry is not None:
            try:
                filter_geom = _shp_shape(geometry)
            except Exception as e:
                raise ValueError(f"Invalid geometry filter: {e}")
            gx0, gy0, gx1, gy1 = filter_geom.bounds
            query_mbr = BoundingBox(minx=gx0, miny=gy0, maxx=gx1, maxy=gy1)
        elif mbr_string:
            try:
                query_mbr = BoundingBox.from_string(mbr_string)
            except ValueError as e:
                raise ValueError(f"Invalid MBR format: {e}")

        dataset_path = self.data_root / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")

        # The streamer performs the spatial filtering, so the handler does not
        # re-filter (output_mbr=None).
        format_lower = format.lower()
        if format_lower == 'csv':
            handler = CSVHandler(output_mbr=None)
        elif format_lower == 'geojson':
            handler = GeoJSONHandler(output_mbr=None)
        else:
            raise ValueError(f"Unsupported format: {format}")

        streamer = FeatureStreamer(dataset_path)
        return streamer.stream_features(query_mbr, handler, filter_geom=filter_geom)
    
    def get_mime_type(self, format: str) -> str:
        """Get MIME type for format."""
        format_lower = format.lower()
        if format_lower == 'csv':
            return 'text/csv'
        elif format_lower == 'geojson':
            return 'application/geo+json'
        return 'application/octet-stream'
