import base64
import copy
import io
import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
from utils import SYSTEM_PROMPT, ANALYSIS_TOOL

load_dotenv(dotenv_path=".env")

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ANALYSIS_REQUEST_TEXT = (
    "Analyze this food image and return the macro breakdown for every dish you identify."
)

def _resize_if_needed(image_bytes: bytes, media_type: str) -> bytes:
    if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
        return image_bytes

    fmt_map = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    img = Image.open(io.BytesIO(image_bytes))
    fmt = fmt_map.get(media_type, img.format or "JPEG")

    scale = 0.9
    while True:
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=fmt, optimize=True)
        data = buf.getvalue()
        if len(data) <= MAX_IMAGE_SIZE_BYTES:
            return data


def analyze_food_claude(image_bytes: bytes, media_type: str, model="claude-sonnet-4-6") -> dict:
    """
    Send the image to Claude and return structured macro data.

    Returns a dict matching the `submit_food_analysis` tool schema:
    {
        "dishes": { "<dish_name>": { kcal, protein, fat, carbs, fiber, sugar, quantity, confidence }, ... },
        "total":  { kcal, protein, fat, carbs, fiber, sugar },
        "overall_confidence": float,
        "notes": str   # optional
    }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = Anthropic(api_key=api_key)
    image_bytes = _resize_if_needed(image_bytes, media_type)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[ANALYSIS_TOOL],
        tool_choice={"type": "any"},  # forces Claude to call the tool
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": ANALYSIS_REQUEST_TEXT,
                    },
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_food_analysis":
            return block.input

    raise RuntimeError("Claude did not return a structured tool-use response.")

def _extract_openai_json(response) -> dict:
    if hasattr(response, "output_parsed") and isinstance(response.output_parsed, dict):
        return response.output_parsed

    output_text = getattr(response, "output_text", None)
    if output_text:
        return json.loads(output_text)

    output = getattr(response, "output", None) or []
    for item in output:
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) == "output_text":
                text = getattr(block, "text", "")
                if text:
                    return json.loads(text)

    raise RuntimeError("OpenAI response did not contain valid JSON output.")


def _openai_strict_schema(schema: dict) -> dict:
    """
    Convert a permissive JSON schema into OpenAI strict-schema compatible form.
    """
    strict = copy.deepcopy(schema)

    def _walk(node):
        if isinstance(node, dict):
            node_type = node.get("type")

            if node_type == "object":
                node.setdefault("additionalProperties", False)
                node.setdefault("properties", {})
                node.setdefault("required", [])

                props = node.get("properties", {})
                original_required = node.get("required", [])
                updated_required = list(original_required)

                if isinstance(props, dict):
                    for key, child in props.items():
                        if key not in original_required:
                            updated_required.append(key)
                            props[key] = {
                                "anyOf": [child, {"type": "null"}]
                                }
                        else:
                            _walk(child)
                
                if updated_required:
                    node["required"] = updated_required

                addl = node.get("additionalProperties")
                if isinstance(addl, dict):
                    _walk(addl)

            elif node_type == "array":
                _walk(node.get("items"))

            for key in ("anyOf", "allOf", "oneOf"):
                for child in node.get(key, []) or []:
                    _walk(child)

        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(strict)
    return strict


def analyze_food_gpt(image_bytes: bytes, media_type: str, model="gpt-5-mini") -> dict:
    """
    Send the image to OpenAI and return structured macro data.

    Returns a dict matching the `submit_food_analysis` tool schema.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    
    image_bytes = _resize_if_needed(image_bytes, media_type)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    strict_schema = _openai_strict_schema(ANALYSIS_TOOL["input_schema"])

    strict_schema = _openai_strict_schema(ANALYSIS_TOOL["input_schema"])

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ANALYSIS_REQUEST_TEXT},
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{b64_image}",
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": ANALYSIS_TOOL["name"],
                "schema": strict_schema,
                "strict": True,
            }
        }
    )
    return _extract_openai_json(response)

def analyze_food_image(image_bytes, media_type, ai_provider="openai"):
    if ai_provider.lower().strip() == "anthropic":
            result = analyze_food_claude(image_bytes, media_type)
            
    elif ai_provider.lower().strip()=="openai":
        result = analyze_food_gpt(image_bytes, media_type)
    
    return result
