from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .factory import LLMFactory
from .provider import LLMProviderError

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = """\
You are a geospatial map styling assistant for an interactive map viewer.

Your job is to turn a user's styling request into structured JSON that can be
used by the application to render or update a map style.

Rules:
- Always respond with ONLY one JSON object.
- No markdown fences.
- No prose before or after the JSON.
- Preserve conversational context from previous turns when possible.
- On follow-up turns, modify the existing style intent rather than starting over
  unless the user clearly asks to change dataset or attribute.
- Keep outputs compact and deterministic.

Your JSON object must contain exactly these top-level keys:
- "assistant_response": string
- "selected_dataset": string
- "selected_attributes": array of strings
- "style_intent": string
- "style": object

The "style" object must contain these keys:
- "geometry_type": string
- "style_type": string
- "target_attribute": string
- "color_theme": object
- "opacity": number
- "stroke_width": number
- "radius": number
- "legend_title": string
- "notes": array of strings

The "color_theme" object must contain:
- "name": string
- "colors": array of strings

Guidance:
- For line datasets such as roads, prefer style_type values like:
  "line-categorical", "line-gradient", or "line-single-color".
- For polygons, use values like:
  "fill-categorical", "fill-gradient", or "fill-single-color".
- For points, use values like:
  "circle-categorical", "circle-gradient", or "circle-single-color".
- Use valid CSS-style hex colors where possible.
- If a field is not relevant, still include it with a sensible neutral value.
"""

_START_PROMPT_TEMPLATE = """\
User request:
{user_query}

Dataset selected for styling:
{dataset}

Compact dataset summary:
{dataset_summary_json}

Preselected attributes from routing (if any):
{selected_attributes_json}

Optional existing style intent from routing (if any):
{style_intent_json}

Generate the best initial structured style response for this request.
"""

_CONTINUE_PROMPT_TEMPLATE = """\
The user is continuing the same styling conversation.

New user request:
{user_query}

Known dataset for this conversation:
{dataset}

Use prior conversation context to update the style. Do not switch datasets or
attributes unless the user explicitly asks to do so.

If optional reminder context is provided below, use it only as a light hint:

Selected attributes hint:
{selected_attributes_json}

Current style hint:
{current_style_json}
"""


@dataclass
class StyleConversationResult:
    assistant_response: str
    selected_dataset: str
    selected_attributes: List[str]
    style_intent: str
    style: Dict[str, Any]
    interaction_id: Optional[str]
    raw_text: str


def start_style_conversation(
    *,
    dataset: str,
    dataset_summary: Dict[str, Any],
    user_query: str,
    selected_attributes: Optional[List[str]] = None,
    style_intent: Optional[str] = None,
    provider_name: str = "gemini",
    temperature: float = 0.2,
) -> StyleConversationResult:
    """Start a stateful style conversation.

    This should be used after your dataset router has already selected the
    dataset, and optionally selected likely styling attributes.
    """
    provider = LLMFactory.get_provider(provider_name)

    prompt = _START_PROMPT_TEMPLATE.format(
        user_query=user_query.strip(),
        dataset=dataset,
        dataset_summary_json=json.dumps(dataset_summary, ensure_ascii=False, indent=2),
        selected_attributes_json=json.dumps(selected_attributes or [], ensure_ascii=False),
        style_intent_json=json.dumps(style_intent, ensure_ascii=False),
    )

    response = provider.generate_response(
        prompt,
        previous_interaction_id=None,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=temperature,
    )
    logger.debug("Initial style conversation raw response: %s", response.text)

    parsed = _parse_style_response(response.text)
    return StyleConversationResult(
        assistant_response=parsed["assistant_response"],
        selected_dataset=parsed["selected_dataset"],
        selected_attributes=parsed["selected_attributes"],
        style_intent=parsed["style_intent"],
        style=parsed["style"],
        interaction_id=response.interaction_id,
        raw_text=response.text,
    )


def continue_style_conversation(
    *,
    dataset: str,
    user_query: str,
    previous_interaction_id: str,
    selected_attributes_hint: Optional[List[str]] = None,
    current_style_hint: Optional[Dict[str, Any]] = None,
    provider_name: str = "gemini",
    temperature: float = 0.2,
) -> StyleConversationResult:
    """Continue a stateful style conversation using a prior interaction id."""
    if not previous_interaction_id:
        raise ValueError("previous_interaction_id is required for continuation")

    provider = LLMFactory.get_provider(provider_name)

    prompt = _CONTINUE_PROMPT_TEMPLATE.format(
        user_query=user_query.strip(),
        dataset=dataset,
        selected_attributes_json=json.dumps(selected_attributes_hint or [], ensure_ascii=False),
        current_style_json=json.dumps(current_style_hint or {}, ensure_ascii=False, indent=2),
    )

    response = provider.generate_response(
        prompt,
        previous_interaction_id=previous_interaction_id,
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=temperature,
    )
    logger.debug("Follow-up style conversation raw response: %s", response.text)

    parsed = _parse_style_response(response.text)
    return StyleConversationResult(
        assistant_response=parsed["assistant_response"],
        selected_dataset=parsed["selected_dataset"],
        selected_attributes=parsed["selected_attributes"],
        style_intent=parsed["style_intent"],
        style=parsed["style"],
        interaction_id=response.interaction_id,
        raw_text=response.text,
    )


def _parse_style_response(text: str) -> Dict[str, Any]:
    """Parse the provider's JSON object and normalize required fields."""
    cleaned = _strip_code_fences(text)
    json_blob = _extract_first_json_object(cleaned)

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in LLM response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got: {type(parsed).__name__}")

    assistant_response = parsed.get("assistant_response")
    selected_dataset = parsed.get("selected_dataset")
    selected_attributes = parsed.get("selected_attributes")
    style_intent = parsed.get("style_intent")
    style = parsed.get("style")

    if not isinstance(assistant_response, str):
        raise ValueError("Response JSON missing string field 'assistant_response'")
    if not isinstance(selected_dataset, str):
        raise ValueError("Response JSON missing string field 'selected_dataset'")
    if not isinstance(selected_attributes, list) or not all(isinstance(x, str) for x in selected_attributes):
        raise ValueError("Response JSON field 'selected_attributes' must be a list[str]")
    if not isinstance(style_intent, str):
        raise ValueError("Response JSON missing string field 'style_intent'")
    if not isinstance(style, dict):
        raise ValueError("Response JSON missing object field 'style'")

    normalized_style = _normalize_style(style)

    return {
        "assistant_response": assistant_response.strip(),
        "selected_dataset": selected_dataset.strip(),
        "selected_attributes": [x.strip() for x in selected_attributes if x.strip()],
        "style_intent": style_intent.strip(),
        "style": normalized_style,
    }


def _normalize_style(style: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required style keys exist with stable defaults."""
    color_theme = style.get("color_theme")
    if not isinstance(color_theme, dict):
        color_theme = {}

    colors = color_theme.get("colors")
    if not isinstance(colors, list) or not all(isinstance(c, str) for c in colors):
        colors = []

    normalized = {
        "geometry_type": _as_string(style.get("geometry_type"), "unknown"),
        "style_type": _as_string(style.get("style_type"), "line-single-color"),
        "target_attribute": _as_string(style.get("target_attribute"), ""),
        "color_theme": {
            "name": _as_string(color_theme.get("name"), "custom"),
            "colors": colors,
        },
        "opacity": _as_float(style.get("opacity"), 1.0),
        "stroke_width": _as_float(style.get("stroke_width"), 1.5),
        "radius": _as_float(style.get("radius"), 4.0),
        "legend_title": _as_string(style.get("legend_title"), ""),
        "notes": _as_string_list(style.get("notes")),
    }
    return normalized


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from text."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    raise ValueError("Unterminated JSON object in LLM response")


def _as_string(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) else default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]