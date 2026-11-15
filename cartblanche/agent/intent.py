"""E2: Intent Parser. Turns the request into Structured Requirements (Constraints).

Brand is hard ("only Dewlab", "a Dewlab serum"), soft ("I like Dewlab") or
none. Filling an empty brand from history as a soft preference needs the
history, so it happens in the personalization step, not here.
"""

from __future__ import annotations

import json

from cartblanche.llm import LLM
from cartblanche.schemas import Constraints

PROMPT = """Extract the shopper's requirements from their request. Only include what the request says; never guess.

Reply with only a JSON object:
{"max_price": number or null,
 "min_rating": number or null,
 "brand": {"name": string or null, "strength": "hard" | "soft" | "none"},
 "required_features": [short phrases]}

brand.strength: "hard" when the shopper names a brand they want ("only Dewlab", "a Dewlab serum");
"soft" when they say they like a brand but don't require it; "none" when no brand is mentioned.
required_features: must-have properties, as short lowercase phrases, e.g. "fragrance-free", "spf 50", "sulfate-free".
Don't list the product type itself (not "moisturizer")."""


def parse_json(text: str | None) -> dict:
    """The first {...} object in a model reply, tolerating ```json fences and surrounding prose."""
    if not text or "{" not in text:
        raise ValueError(f"no JSON object in reply: {text!r}")
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


def parse_intent(request: str, llm: LLM, tag: str = "") -> Constraints:
    reply = llm.complete(
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": request}], tag=f"{tag}/intent"
    )
    fields = {k: v for k, v in parse_json(reply.content).items() if v is not None}  # null means "not stated"
    return Constraints(**fields)
