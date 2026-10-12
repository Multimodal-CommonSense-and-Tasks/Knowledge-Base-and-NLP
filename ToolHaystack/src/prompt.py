"""Prompt construction: API documents + long-term dialogue -> next API call."""

import json

SYSTEM_PROMPT = """You are a tool-augmented assistant that has been talking with the same user across many sessions.
You are given a set of APIs and the full interaction history. Decide the API call the assistant should make next to fulfil the user's latest request.

Rules:
- Choose exactly one API from the provided list and fill in its parameters.
- Take argument values from the interaction history. If the user refers to something said earlier, resolve it from the history; if the user has changed or updated a value, use the most recent one.
- If the value of a required parameter was never provided in the history, do not guess or use a default value; set it to "MISSING".

Respond with a single JSON object and nothing else:
{"api_name": "<API name>", "parameters": {"<parameter name>": "<value>", ...}}"""


def format_api(doc):
    def params(ps):
        return [
            {"name": p["name"], "type": p.get("type", ""), "description": p.get("description", "")}
            for p in ps or []
        ]

    return {
        "api_name": doc["api_name"],
        "description": doc.get("api_desc") or "",
        "tool": doc.get("tool_name") or "",
        "tool_description": doc.get("tool_desc") or "",
        "required_parameters": params(doc.get("required_params")),
        "optional_parameters": params(doc.get("optional_params")),
    }


def build_messages(sample):
    apis = "\n".join(json.dumps(format_api(d), ensure_ascii=False) for d in sample.candidates)
    user = (
        f"## Available APIs\n{apis}\n\n"
        f"## Interaction history\n{sample.dialogue}\n\n"
        "## Task\nOutput the JSON for the API call that should be made now."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
