"""Flask application factory for the Starlet tile server."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import os
import re

from flask import Flask, Response, render_template, send_from_directory, request
from flask_cors import CORS

from .tiler.tiler import VectorTiler
from .download_service import DatasetFeatureService
from .catalog.embedder import GeminiTextEmbedder
from .catalog.index import CATALOG_FILENAME
from .catalog.router import CatalogRouter, SearchBackend
from .catalog.pgvector_store import PgVectorConfig, PgVectorStore
from .llm import start_style_conversation, continue_style_conversation

logger = logging.getLogger(__name__)


def create_app(
    data_dir: str,
    cache_size: int = 256,
    log_level: Optional[str] = None,
) -> Flask:
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    template_dir = str(Path(__file__).parent / "templates")
    app = Flask(__name__, template_folder=template_dir)
    CORS(app, resources={r"/*": {"origins": "*"}})

    data_root = Path(data_dir)
    tiler_cache: Dict[str, VectorTiler] = {}
    feature_service = DatasetFeatureService(data_root)

    _catalog_runtime: Dict[str, Any] = {
        "router": None,
        "mtime": None,
    }

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    def get_tiler(dataset: str) -> VectorTiler:
        if dataset not in tiler_cache:
            tiler_cache[dataset] = VectorTiler(
                str(data_root / dataset),
                memory_cache_size=cache_size,
            )
        return tiler_cache[dataset]

    def _catalog_index_path() -> Path:
        return data_root / "_catalog" / CATALOG_FILENAME

    def _catalog_backend_from_env() -> SearchBackend:
        value = os.environ.get("CATALOG_SEARCH_BACKEND", "auto").strip().lower()
        if value == "pgvector":
            return SearchBackend.PGVECTOR
        if value == "npy":
            return SearchBackend.NPY
        return SearchBackend.AUTO

    def _build_catalog_router() -> CatalogRouter:
        index_path = _catalog_index_path()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Catalogue index not found at {index_path}. "
                f"Build it first with catalog/index.py."
            )

        embedder = GeminiTextEmbedder()
        backend = _catalog_backend_from_env()

        pg_store = None
        if backend in (SearchBackend.AUTO, SearchBackend.PGVECTOR):
            pg_store = PgVectorStore(PgVectorConfig())

        return CatalogRouter(
            index_dir_or_file=str(index_path.parent),
            embedder=embedder,
            backend=backend,
            pgvector_store=pg_store,
        )

    def _get_catalog_router() -> CatalogRouter:
        index_path = _catalog_index_path()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Catalogue index not found at {index_path}. "
                f"Build it first with catalog/index.py."
            )

        mtime = index_path.stat().st_mtime
        if _catalog_runtime["router"] is None or _catalog_runtime["mtime"] != mtime:
            _catalog_runtime["router"] = _build_catalog_router()
            _catalog_runtime["mtime"] = mtime
        return _catalog_runtime["router"]

    def _load_stats_for_dataset(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        stats_path = dataset_path / "stats" / "attributes.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats not found for dataset: {dataset}")

        with open(stats_path, "r") as f:
            return json.load(f)

    def _dataset_exists(dataset: str) -> bool:
        path = data_root / dataset
        return path.exists() and path.is_dir()

    def _dataset_metadata(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        return {
            "id": dataset,
            "name": dataset.replace("_", " ").title(),
            "size": sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()),
            "file_count": sum(1 for f in dataset_path.rglob("*") if f.is_file()),
        }

    def _list_dataset_metadata(query: Optional[str] = None) -> List[Dict[str, Any]]:
        datasets: List[Dict[str, Any]] = []
        if not data_root.exists():
            return datasets

        query_lc = (query or "").strip().lower()

        for d in sorted(data_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("."):
                continue
            if d.name == "_catalog":
                continue

            item = {
                "id": d.name,
                "name": d.name.replace("_", " ").title(),
                "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
            }

            if not query_lc or query_lc in item["id"].lower() or query_lc in item["name"].lower():
                datasets.append(item)

        return datasets

    def _extract_first_json_array(text: str) -> List[Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in LLM response")
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array from LLM response")
        return parsed

    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < 0 or end < start:
            raise ValueError("No JSON object found in LLM response")
        parsed = json.loads(cleaned[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object from LLM response")
        return parsed

    # -------------------------------------------------------------------------
    # Styling helpers
    # -------------------------------------------------------------------------

    def _infer_geometry_kind_from_summary(summary: Dict[str, Any]) -> str:
        geometry = summary.get("geometry") or []
        geom_types: List[str] = []

        for geom_attr in geometry:
            stats = geom_attr.get("geom_types") or {}
            geom_types.extend(str(k).lower() for k in stats.keys())

        joined = " ".join(geom_types)
        if any(x in joined for x in ("line", "multiline")):
            return "line"
        if any(x in joined for x in ("polygon", "multipolygon")):
            return "polygon"
        if any(x in joined for x in ("point", "multipoint")):
            return "point"

        dataset_name = str(summary.get("dataset", "")).lower()
        if "road" in dataset_name or "rail" in dataset_name:
            return "line"
        if "county" in dataset_name or "state" in dataset_name or "tract" in dataset_name:
            return "polygon"
        if "point" in dataset_name:
            return "point"
        return "unknown"

    def _attribute_role_from_summary(attr: Dict[str, Any]) -> str:
        role = str(attr.get("role", "")).strip().lower()
        if role:
            return role

        if attr.get("min") is not None or attr.get("max") is not None:
            return "numeric"
        if attr.get("top_k"):
            return "categorical"
        return "unknown"

    def _find_attribute_summary(summary: Dict[str, Any], attr_name: str) -> Optional[Dict[str, Any]]:
        for attr in summary.get("attributes") or []:
            if str(attr.get("name")) == attr_name:
                return attr
        return None

    def _normalize_hex_color(color: str, fallback: str) -> str:
        color = str(color or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            return color
        return fallback

    def _categorical_palette() -> List[str]:
        return [
            "#1f78b4",
            "#33a02c",
            "#e31a1c",
            "#ff7f00",
            "#6a3d9a",
            "#b15928",
            "#a6cee3",
            "#b2df8a",
        ]

    def _gradient_palette(default_name: str, colors: List[str]) -> Dict[str, Any]:
        sane = [_normalize_hex_color(c, "#4682B4") for c in (colors or [])]
        sane = [c for c in sane if c]
        if not sane:
            sane = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        return {
            "name": default_name,
            "colors": sane,
        }

    def _normalize_style_for_client(
        dataset: str,
        dataset_summary: Dict[str, Any],
        style: Dict[str, Any],
    ) -> Dict[str, Any]:
        geometry_kind = _infer_geometry_kind_from_summary(dataset_summary)
        style_type = str(style.get("style_type", "")).strip() or (
            "line-single-color" if geometry_kind == "line"
            else "fill-single-color" if geometry_kind == "polygon"
            else "circle-single-color" if geometry_kind == "point"
            else "line-single-color"
        )

        target_attribute = str(style.get("target_attribute", "")).strip()
        attr_summary = _find_attribute_summary(dataset_summary, target_attribute) if target_attribute else None
        attr_role = _attribute_role_from_summary(attr_summary or {})
        is_categorical = "categorical" in style_type or attr_role in {"categorical", "categorical_text"}
        is_gradient = "gradient" in style_type

        color_theme = style.get("color_theme") or {}
        theme_name = str(color_theme.get("name", "")).strip() or "custom"
        theme_colors = color_theme.get("colors") or []

        opacity = float(style.get("opacity", 1.0) or 1.0)
        stroke_width = float(style.get("stroke_width", 2.0) or 2.0)
        radius = float(style.get("radius", 4.0) or 4.0)
        legend_title = str(style.get("legend_title", "")).strip() or target_attribute or dataset
        notes = style.get("notes") or []
        if not isinstance(notes, list):
            notes = [str(notes)]

        if is_categorical:
            palette = [_normalize_hex_color(c, "#1f78b4") for c in theme_colors]
            palette = [c for c in palette if c]
            if not palette:
                palette = _categorical_palette()

            categorical_values: List[str] = []
            if attr_summary:
                for item in attr_summary.get("top_k") or []:
                    if isinstance(item, dict):
                        value = item.get("value")
                    else:
                        value = item
                    if value is None:
                        continue
                    categorical_values.append(str(value))

            stops = [
                {"value": value, "color": palette[i % len(palette)]}
                for i, value in enumerate(categorical_values)
            ]

            return {
                "dataset": dataset,
                "geometry_kind": geometry_kind,
                "style_type": style_type,
                "target_attribute": target_attribute,
                "legend_title": legend_title,
                "opacity": opacity,
                "stroke_width": stroke_width,
                "radius": radius,
                "color_theme": {
                    "name": theme_name,
                    "colors": palette,
                },
                "renderer": {
                    "mode": "categorical",
                    "attribute": target_attribute,
                    "fallback_color": palette[0],
                    "stops": stops,
                },
                "notes": [str(n) for n in notes],
            }

        if is_gradient:
            palette = _gradient_palette(theme_name or "gradient", list(theme_colors))
            min_value = None
            max_value = None
            if attr_summary:
                min_value = attr_summary.get("min")
                max_value = attr_summary.get("max")

            try:
                min_value = float(min_value) if min_value is not None else 0.0
            except Exception:
                min_value = 0.0
            try:
                max_value = float(max_value) if max_value is not None else 1.0
            except Exception:
                max_value = 1.0
            if max_value == min_value:
                max_value = min_value + 1.0

            return {
                "dataset": dataset,
                "geometry_kind": geometry_kind,
                "style_type": style_type,
                "target_attribute": target_attribute,
                "legend_title": legend_title,
                "opacity": opacity,
                "stroke_width": stroke_width,
                "radius": radius,
                "color_theme": palette,
                "renderer": {
                    "mode": "gradient",
                    "attribute": target_attribute,
                    "min": min_value,
                    "max": max_value,
                    "colors": palette["colors"],
                },
                "notes": [str(n) for n in notes],
            }

        fallback_color = "#4682B4"
        if theme_colors:
            fallback_color = _normalize_hex_color(theme_colors[0], fallback_color)

        return {
            "dataset": dataset,
            "geometry_kind": geometry_kind,
            "style_type": style_type,
            "target_attribute": target_attribute,
            "legend_title": legend_title,
            "opacity": opacity,
            "stroke_width": stroke_width,
            "radius": radius,
            "color_theme": {
                "name": theme_name,
                "colors": [fallback_color],
            },
            "renderer": {
                "mode": "single",
                "attribute": target_attribute,
                "color": fallback_color,
            },
            "notes": [str(n) for n in notes],
        }

    def _candidate_payload_for_llm(candidates) -> List[Dict[str, Any]]:
        payload = []
        for c in candidates:
            payload.append({
                "dataset": c.dataset,
                "score": round(float(c.score), 6),
                "summary": c.summary,
            })
        return payload

    def _looks_like_new_dataset_request(
        user_query: str,
        current_dataset: Optional[str],
        current_style: Optional[Dict[str, Any]],
    ) -> bool:
        q = (user_query or "").strip().lower()
        if not q:
            return False

        if any(
            phrase in q for phrase in [
                "use a different dataset",
                "switch dataset",
                "another dataset",
                "different dataset",
                "new dataset",
            ]
        ):
            return True

        if current_dataset and current_dataset.lower() in q:
            return False

        domain_triggers = [
            "county", "counties", "state", "states", "tract", "tracts",
            "road", "roads", "rail", "rails", "building", "buildings",
            "point", "points", "landmark", "landmarks",
        ]
        if any(word in q for word in domain_triggers):
            target_attr = ""
            if current_style:
                target_attr = str(current_style.get("target_attribute", "")).lower()
            current_dataset_lc = (current_dataset or "").lower()
            if current_dataset_lc and current_dataset_lc not in q and target_attr and target_attr not in q:
                return True

        return False

    def _run_initial_chat_turn(user_query: str, k: int = 5) -> Tuple[Dict[str, Any], int]:
        router = _get_catalog_router()
        candidates = router.search(user_query, k=k)
        if not candidates:
            raise LookupError("No indexed datasets available")

        candidate_payload = _candidate_payload_for_llm(candidates)

        turn1 = start_style_conversation(
            dataset="__choose_from_candidates__",
            dataset_summary={
                "mode": "candidate_selection",
                "candidates": candidate_payload,
            },
            user_query=(
                f"{user_query}\n\n"
                "Choose the single best dataset from the provided candidates, "
                "choose the best attribute for styling, and generate the initial style."
            ),
            selected_attributes=None,
            style_intent=None,
            provider_name="gemini",
            temperature=0.2,
        )

        selected_dataset = str(turn1.selected_dataset).strip()
        candidate_by_name = {c.dataset: c for c in candidates}
        if selected_dataset not in candidate_by_name:
            logger.warning(
                "[ChatStyle] LLM selected invalid dataset '%s'; falling back to top candidate '%s'",
                selected_dataset,
                candidates[0].dataset,
            )
            selected_dataset = candidates[0].dataset

        selected_candidate = candidate_by_name[selected_dataset]
        selected_summary = selected_candidate.summary
        normalized_style = _normalize_style_for_client(
            dataset=selected_dataset,
            dataset_summary=selected_summary,
            style=turn1.style,
        )

        response = {
            "mode": "initial",
            "query": user_query,
            "interaction_id": turn1.interaction_id,
            "assistant_response": turn1.assistant_response,
            "selected_dataset": selected_dataset,
            "selected_dataset_score": float(selected_candidate.score),
            "selected_attributes": turn1.selected_attributes,
            "style_intent": turn1.style_intent,
            "style": normalized_style,
            "top_k": candidate_payload,
        }
        return response, 200

    def _run_followup_chat_turn(
        *,
        user_query: str,
        interaction_id: str,
        current_dataset: str,
        current_attributes: Optional[List[str]] = None,
        current_style: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        if not current_dataset:
            raise ValueError("current_dataset is required for follow-up turns")
        if not interaction_id:
            raise ValueError("interaction_id is required for follow-up turns")

        stats = _load_stats_for_dataset(current_dataset)

        selected_summary = {
            "dataset": current_dataset,
            "geometry": [],
            "attributes": [],
        }

        for attr in stats.get("attributes") or []:
            if not isinstance(attr, dict):
                continue
            stats_obj = attr.get("stats") or {}
            name = attr.get("name")
            entry: Dict[str, Any] = {"name": name}

            if "geom_types" in stats_obj:
                entry["geom_types"] = stats_obj.get("geom_types") or {}
                selected_summary["geometry"].append(entry)
                continue

            if any(k in stats_obj for k in ("min", "max", "mean", "stddev")):
                entry["role"] = "numeric"
                entry["min"] = stats_obj.get("min")
                entry["max"] = stats_obj.get("max")
                entry["top_k"] = stats_obj.get("top_k") or []
            elif stats_obj.get("top_k") is not None:
                entry["role"] = "categorical"
                entry["top_k"] = stats_obj.get("top_k") or []
            else:
                entry["role"] = "unknown"
            selected_summary["attributes"].append(entry)

        turn = continue_style_conversation(
            dataset=current_dataset,
            user_query=user_query,
            previous_interaction_id=interaction_id,
            selected_attributes_hint=current_attributes or [],
            current_style_hint=current_style or {},
            provider_name="gemini",
            temperature=0.2,
        )

        returned_dataset = str(turn.selected_dataset).strip() or current_dataset
        if returned_dataset != current_dataset:
            logger.info(
                "[ChatStyle] Follow-up requested dataset switch from '%s' to '%s'; restarting retrieval.",
                current_dataset,
                returned_dataset,
            )
            return _run_initial_chat_turn(user_query=user_query, k=5)

        normalized_style = _normalize_style_for_client(
            dataset=current_dataset,
            dataset_summary=selected_summary,
            style=turn.style,
        )

        response = {
            "mode": "followup",
            "query": user_query,
            "interaction_id": turn.interaction_id,
            "assistant_response": turn.assistant_response,
            "selected_dataset": current_dataset,
            "selected_attributes": turn.selected_attributes,
            "style_intent": turn.style_intent,
            "style": normalized_style,
            "top_k": [],
        }
        return response, 200

    # -------------------------------------------------------------------------
    # Routes: tiles and dataset files
    # -------------------------------------------------------------------------

    @app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_tile(dataset, z, x, y):
        t0 = perf_counter()
        tiler = get_tiler(dataset)
        data = tiler.get_tile(z, x, y)
        elapsed_ms = (perf_counter() - t0) * 1000
        logger.info(
            "[TileRequest] dataset=%s z=%d x=%d y=%d bytes=%d elapsed=%.1fms",
            dataset, z, x, y, len(data), elapsed_ms,
        )
        return Response(data, mimetype="application/vnd.mapbox-vector-tile")

    @app.get("/api/datasets")
    def list_datasets():
        datasets = sorted([d.name for d in data_root.iterdir()]) if data_root.exists() else []
        return app.response_class(
            response=json.dumps({"datasets": datasets}),
            mimetype="application/json",
        )

    @app.get("/datasets.json")
    def search_datasets():
        query = request.args.get("q", default=None)
        datasets = _list_dataset_metadata(query=query)
        return app.response_class(
            response=json.dumps({"datasets": datasets}, indent=2),
            mimetype="application/json",
        )

    @app.get("/datasets/<dataset>.json")
    def get_dataset_metadata(dataset):
        try:
            metadata = _dataset_metadata(dataset)
            return app.response_class(
                response=json.dumps(metadata, indent=2),
                mimetype="application/json",
            )
        except FileNotFoundError:
            return {"error": "Dataset not found"}, 404
        except Exception as e:
            return {"error": f"Failed to retrieve metadata: {str(e)}"}, 500

    @app.get("/api/datasets/<dataset>/stats")
    def get_dataset_stats(dataset):
        stats_path = data_root / dataset / "stats" / "attributes.json"
        if not stats_path.exists():
            return {"error": "Stats not found for dataset"}, 404
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {str(e)}"}, 500

    @app.get("/datasets/<dataset>.html")
    def visualize_dataset(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return "<h1>Dataset not found</h1>", 404
        try:
            return render_template(
                "view_dataset.html",
                dataset_id=dataset,
                dataset_name=dataset.replace("_", " ").title(),
            )
        except Exception as e:
            return f"<h1>Failed to render visualization: {str(e)}</h1>", 500

    @app.get("/datasets/<dataset>/features.<format>")
    def download_features(dataset, format):
        try:
            mbr_string = request.args.get("mbr", default=None)
            feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            if mbr_string:
                filename = f"{dataset}_{mbr_string.replace(',', '_')}.{format}"
            else:
                filename = f"{dataset}_full.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.post("/datasets/<dataset>/features.<format>")
    def download_features_with_geometry(dataset, format):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            geojson_payload = request.get_json()
            mbr_string = request.args.get("mbr", default=None)
            if geojson_payload:
                geometry = geojson_payload.get("geometry")
                if not geometry:
                    return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400
                feature_stream = feature_service.get_features_stream(
                    dataset,
                    format,
                    geometry=geometry,
                )
            else:
                feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.get("/datasets/<dataset>/features/sample.json")
    def get_sample_non_geometry_attributes(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=False)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return app.response_class(
                response=json.dumps(sample_record, indent=2),
                mimetype="application/json",
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.get("/datasets/<dataset>/features/sample.geojson")
    def get_sample_with_geometry(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=True)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return app.response_class(
                response=json.dumps(sample_record, indent=2),
                mimetype="application/json",
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    # -------------------------------------------------------------------------
    # Routes: UI
    # -------------------------------------------------------------------------

    @app.get("/")
    def index():
        logger.info("Serving index page")
        return render_template("index.html")

    @app.route("/<path:filename>")
    def serve_file(filename):
        server_dir = Path(__file__).parent
        file_path = server_dir / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(server_dir), filename)
        return "File not found", 404

    # -------------------------------------------------------------------------
    # Routes: conversational LLM styling
    # -------------------------------------------------------------------------

    @app.post("/api/chat-style")
    def chat_style():
        json_ct = {"Content-Type": "application/json"}
        body = request.get_json(silent=True) or {}

        user_query = str(body.get("query", "")).strip()
        if not user_query:
            return json.dumps({"error": "Request body must include a non-empty 'query'"}), 400, json_ct

        interaction_id = str(body.get("interaction_id", "") or "").strip()
        current_dataset = str(body.get("current_dataset", "") or "").strip()

        current_attributes_raw = body.get("current_attributes")
        if isinstance(current_attributes_raw, list):
            current_attributes = [str(x) for x in current_attributes_raw if isinstance(x, (str, int, float))]
        else:
            current_attributes = []

        current_style = body.get("current_style")
        if not isinstance(current_style, dict):
            current_style = {}

        requested_k = body.get("k", 5)
        try:
            k = max(1, min(int(requested_k), 10))
        except Exception:
            k = 5

        try:
            # Home page starts with no dataset selected.
            # First turn: retrieve candidates and let the LLM choose the best dataset,
            # attribute, and style.
            if not interaction_id or not current_dataset:
                response, status = _run_initial_chat_turn(user_query=user_query, k=k)
                return json.dumps(response, indent=2), status, json_ct

            # Follow-up turns continue the current interaction on the current dataset,
            # unless the prompt clearly requests a different dataset/domain.
            if _looks_like_new_dataset_request(
                user_query=user_query,
                current_dataset=current_dataset,
                current_style=current_style,
            ):
                response, status = _run_initial_chat_turn(user_query=user_query, k=k)
                return json.dumps(response, indent=2), status, json_ct

            if not _dataset_exists(current_dataset):
                return json.dumps({"error": f"Current dataset not found: {current_dataset}"}), 404, json_ct

            response, status = _run_followup_chat_turn(
                user_query=user_query,
                interaction_id=interaction_id,
                current_dataset=current_dataset,
                current_attributes=current_attributes,
                current_style=current_style,
            )
            return json.dumps(response, indent=2), status, json_ct

        except FileNotFoundError as e:
            return json.dumps({"error": str(e)}), 503, json_ct
        except LookupError as e:
            return json.dumps({"error": str(e)}), 404, json_ct
        except Exception as e:
            logger.exception("[ChatStyle] Failed")
            return json.dumps({"error": f"Chat styling failed: {e}"}), 500, json_ct

    # -------------------------------------------------------------------------
    # Compatibility route: keep old endpoint name but route into chat API.
    # -------------------------------------------------------------------------

    @app.post("/api/query-styles")
    def query_styles():
        body = request.get_json(silent=True) or {}
        with app.test_request_context(
            "/api/chat-style",
            method="POST",
            json=body,
        ):
            return chat_style()

    return app