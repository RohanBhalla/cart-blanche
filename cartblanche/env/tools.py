"""Store tools in OpenAI function-calling format (what LiteLLM expects), plus dispatch."""

from __future__ import annotations

import json
from typing import Any

from cartblanche.env.store import TaskStore

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the store. Returns short product summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. 'running earbuds'"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "max_price": {"type": "number"},
                            "min_price": {"type": "number"},
                            "min_rating": {"type": "number"},
                            "brand": {"type": "string"},
                            "category": {"type": "string"},
                        },
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Full product page: price, features, description, details.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": "Customer reviews for a product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": "The user's past purchases with their ratings and reviews.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_owned",
            "description": "Past purchases matching a category or item type, e.g. 'air fryer'.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}, "category": {"type": "string"}},
                "required": ["user_id", "category"],
            },
        },
    },
]

TOOL_NAMES = {spec["function"]["name"] for spec in TOOL_SPECS}


def dispatch(store: TaskStore, name: str, arguments: str | dict[str, Any]) -> str:
    """Run one tool call and return its JSON result. Bad calls return an error the LLM can read."""
    if name not in TOOL_NAMES:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        result = getattr(store, name)(**args)
    except (json.JSONDecodeError, TypeError) as e:
        result = {"error": f"bad arguments for {name}: {e}"}
    return json.dumps(result)
