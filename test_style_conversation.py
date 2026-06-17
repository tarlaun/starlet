#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from starlet._internal.server.llm import start_style_conversation, continue_style_conversation


def pretty(title: str, obj) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(obj)


def print_turn(turn_num: int, result) -> None:
    pretty(f"TURN {turn_num} ASSISTANT RESPONSE", result.assistant_response)
    pretty(f"TURN {turn_num} SELECTED DATASET", result.selected_dataset)
    pretty(f"TURN {turn_num} SELECTED ATTRIBUTES", result.selected_attributes)
    pretty(f"TURN {turn_num} STYLE INTENT", result.style_intent)
    pretty(f"TURN {turn_num} STYLE JSON", result.style)
    pretty(f"TURN {turn_num} INTERACTION ID", result.interaction_id)
    pretty(f"TURN {turn_num} RAW TEXT", result.raw_text)


def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")

    dataset = "ne_roads"
    dataset_summary = {
        "dataset_name": "ne_roads",
        "geometry_type": "LineString",
        "description": "Road segments with attributes such as speed limit, road class, and name.",
        "columns": [
            {
                "name": "speed_limit",
                "type": "integer",
                "description": "Posted speed limit for the road segment",
                "sample_values": [25, 35, 45, 55, 65, 70],
            },
            {
                "name": "road_class",
                "type": "string",
                "description": "Functional road class",
                "sample_values": ["local", "secondary", "primary", "highway"],
            },
            {
                "name": "name",
                "type": "string",
                "description": "Road name",
                "sample_values": ["Main St", "I-10", "University Ave"],
            },
        ],
    }

    prompts = [
        "color the roads based on speed limit",
        "change the color theme to be red to green instead",
        "make the lines thinner",
        "actually color by road_class instead",
    ]

    # -------------------------
    # TURN 1
    # -------------------------
    pretty("TURN 1 USER PROMPT", prompts[0])

    turn1 = start_style_conversation(
        dataset=dataset,
        dataset_summary=dataset_summary,
        user_query=prompts[0],
        selected_attributes=["speed_limit"],
        style_intent="Color roads by speed limit",
        provider_name="gemini",
        temperature=0.2,
    )

    print_turn(1, turn1)

    if not turn1.interaction_id:
        raise RuntimeError("No interaction_id returned from turn 1")

    # -------------------------
    # TURN 2
    # -------------------------
    pretty("TURN 2 USER PROMPT", prompts[1])

    turn2 = continue_style_conversation(
        dataset=dataset,
        user_query=prompts[1],
        previous_interaction_id=turn1.interaction_id,
        selected_attributes_hint=turn1.selected_attributes,
        current_style_hint=turn1.style,
        provider_name="gemini",
        temperature=0.2,
    )

    print_turn(2, turn2)

    if not turn2.interaction_id:
        raise RuntimeError("No interaction_id returned from turn 2")

    # -------------------------
    # TURN 3
    # -------------------------
    pretty("TURN 3 USER PROMPT", prompts[2])

    turn3 = continue_style_conversation(
        dataset=dataset,
        user_query=prompts[2],
        previous_interaction_id=turn2.interaction_id,
        selected_attributes_hint=turn2.selected_attributes,
        current_style_hint=turn2.style,
        provider_name="gemini",
        temperature=0.2,
    )

    print_turn(3, turn3)

    if not turn3.interaction_id:
        raise RuntimeError("No interaction_id returned from turn 3")

    # -------------------------
    # TURN 4
    # -------------------------
    pretty("TURN 4 USER PROMPT", prompts[3])

    turn4 = continue_style_conversation(
        dataset=dataset,
        user_query=prompts[3],
        previous_interaction_id=turn3.interaction_id,
        selected_attributes_hint=turn3.selected_attributes,
        current_style_hint=turn3.style,
        provider_name="gemini",
        temperature=0.2,
    )

    print_turn(4, turn4)

    if not turn4.interaction_id:
        raise RuntimeError("No interaction_id returned from turn 4")

    # -------------------------
    # FINAL SUMMARY CHECK
    # -------------------------
    pretty("FINAL SUMMARY", {
        "turn1_attribute": turn1.style.get("target_attribute"),
        "turn2_attribute": turn2.style.get("target_attribute"),
        "turn3_attribute": turn3.style.get("target_attribute"),
        "turn4_attribute": turn4.style.get("target_attribute"),
        "turn1_style_type": turn1.style.get("style_type"),
        "turn2_style_type": turn2.style.get("style_type"),
        "turn3_style_type": turn3.style.get("style_type"),
        "turn4_style_type": turn4.style.get("style_type"),
        "turn2_colors": turn2.style.get("color_theme", {}).get("colors"),
        "turn3_stroke_width": turn3.style.get("stroke_width"),
        "turn4_colors": turn4.style.get("color_theme", {}).get("colors"),
    })


if __name__ == "__main__":
    main()