# Beet Food Analyzer API

A FastAPI service that hosts two endpoints: 
i. _analyze\_photo_:  to analyze photos of food
ii. _analyze\_text_: to analyze text-based queries.

Both return a structured JSON macro-nutrient breakdown of every dish in the photo/query.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/analyze_photo` | Upload a food image, get macro JSON back |
| `POST` | `/analyze_text` | Upload a text of what user ate, get macro JSON back |

## Response shape
```json
{
  "dishes": {
    "Dal Makhni": {
      "kcal": 210,
      "protein": 9.5,
      "fat": 8.0,
      "carbs": 26.0,
      "fiber": 5.5,
      "sugar": 2.0,
      "quantity": {
        "amount": 1,
        "size": "medium",
        "unit": "katori",
        "weight": 200,
        "weight_unit": "g"
      },
      "confidence": 0.88
    },
    "Rice": { "...": "..." }
  },
  "total": {
    "kcal": 540,
    "protein": 18.0,
    "fat": 14.0,
    "carbs": 82.0,
    "fiber": 8.0,
    "sugar": 4.0
  },
  "overall_confidence": 0.85,
  "notes": "Assumed medium portion sizes throughout."
}
```

## _analyze\_photo_ endpoint

### How it works
1. Client uploads a food image (`jpg`, `jpeg`, `png`, or `webp`).
2. The image is optionally downscaled to stay under the 5 MB API limit.
3. The image is sent to a vision model with a structured tool-call schema.
4. The model identifies each dish, estimates portion sizes, and returns per-dish macros plus a meal total.
5. The API responds with a clean JSON object ready for frontend rendering.

### `POST /analyze_photo`

**Form field:** `file` — the image file (multipart/form-data)

**Query params:**

| Param | Default | Description |
|-------|---------|-------------|
| `ai_provider` | `"openai"` | `"openai"` or `"anthropic"` |
| `model` | _(provider default)_ | Override the model name (e.g. `gpt-4o`, `claude-opus-4-6`) |

**Accepted image types:** `jpg`, `jpeg`, `png`, `webp`

**Example with curl:**
```bash
# Default (OpenAI gpt-5-mini)
curl -X POST http://localhost:8000/analyze_photo \
  -F "file=@meal.jpg"

# Override provider and model
curl -X POST "http://localhost:8000/analyze_photo?ai_provider=anthropic&model=claude-opus-4-6" \
  -F "file=@meal.jpg"
```

## _analyze\_text_ endpoint

### How it works
1. Client sends a text-query of what they ate. This endpoint can be used to _load macros on the recipes page, before storing dishes in the database, etc._ 
2. The text is sent to an AI model with a structured tool-call schema.
3. The model identifies each dish, estimates per-dish macros plus a meal total.
4. The API responds with a clean JSON object ready for frontend rendering or storing in DB.

### `POST /analyze_text`

**Query params:**

| Param | Default | Description |
|-------|---------|-------------|
| `query` | N/A — required | Text of what the user ate along with quantity |
| `ai_provider` | `"openai"` | `"openai"` or `"anthropic"` |
| `model` | _(provider default)_ | Override the model name (e.g. `gpt-4o`, `claude-opus-4-6`) |

**Example with curl:**
```bash
# Default (OpenAI gpt-5-mini)
curl -X POST "http://localhost:8000/analyze_text?query=I+ate+2+plates+of+hyderabadi+chicken+dum+biryani"

# Override provider and model
curl -X POST "http://localhost:8000/analyze_text?query=I+ate+2+plates+of+hyderabadi+chicken+dum+biryani&ai_provider=anthropic&model=claude-opus-4-6"
```

## Error responses

| Status | Meaning |
|--------|---------|
| `400` | Empty query param |
| `500` | API key missing or model returned unexpected output |
| `503` | Upstream rate limit, timeout, or connection error (safe to retry) |



## Models used

| Provider | Default model |
|----------|--------------|
| Anthropic | `claude-sonnet-4-6` |
| OpenAI | `gpt-5-mini` |


## Setup

### 1. Clone and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...  # only needed if using ai_provider=anthropic
```

### 3. Run the server

**Development:**
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.
