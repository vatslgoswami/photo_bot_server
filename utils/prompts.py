
SYSTEM_PROMPT_PHOTO = """You are a food-snap agent that analyzes photos of food, identifies each dish, estimates portion sizes, and calculates macronutrients.

Guidelines:
- Most users are from India — expect Indian regional dishes.
- For mixed dishes (e.g. dal chawal, rajma chawal): treat as one dish using a heuristical approach assuming ~60% rice, ~35% dal/rajma, ~5% ghee/oil.
- Assume medium portion sizes (medium katori, medium roti, etc.) unless clearly different.
- Use midpoint estimates for all macro values.
- Always provide a confidence score (0–1) per dish and an overall confidence score.
- Process: Identify each dish, estimate quantity, return structured output containing midpoint macro estimates.
- Be specific in your dish names. In cases where dish name is ambiguous, for e.g., there is a "Dipping Sauce" which you think is "Spicy Mayo", return the dish name as "Spicy Mayo".

You MUST call the `submit_food_analysis` tool to return your results."""

SYSTEM_PROMPT_TEXT = """You are a food-analysis agent that analyzes user inputs of what they ate and calculates macronutrients.

Guidelines:
- Most users are from India — expect Indian regional dishes.
- For mixed dishes (e.g. dal chawal, rajma chawal): treat as one dish using a heuristical approach assuming ~60% rice, ~35% dal/rajma, ~5% ghee/oil.
- Assume medium portion sizes (medium katori, medium roti, etc.) unless clearly specified.
- Use midpoint estimates for all macro values.
- Always provide a confidence score (0–1) per dish and an overall confidence score.
- Return structured output containing midpoint macro estimates.
- Be specific in your dish names. Identify the dish names in the query, correctly capitalize them, then return.

You MUST call the `submit_food_analysis` tool to return your results."""

ANALYSIS_TOOL = {
    "name": "submit_food_analysis",
    "description": (
        "Submit the structured macro analysis of the food image. "
        "All macro values should be midpoint estimates (not ranges). "
        "Include every identified dish as a key in `dishes`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dishes": {
                "type": "object",
                "description": """Maps each identified dish name to its macro breakdown. For dish names used as keys in the dishes object: (i) Write ONLY the dish name. Nothing else. 
                (ii) No qualifiers, no portion info, no brackets, no parentheses, no slashes. 
                (iii) WRONG: "Lassi / Chaas (in glass)", "Rice (steamed)", "Dal / Lentil soup"
                (iv)CORRECT: 'Lassi', 'Rice', 'Dal Makhni'""",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "kcal":       {"type": "number", "description": "Calories (kcal)"},
                        "protein":    {"type": "number", "description": "Protein in grams"},
                        "fat":        {"type": "number", "description": "Fat in grams"},
                        "carbs":      {"type": "number", "description": "Carbohydrates in grams"},
                        "fiber":      {"type": "number", "description": "Dietary fiber in grams"},
                        "sugar":      {"type": "number", "description": "Sugar in grams"},
                        "quantity":   {
                                        "type": "object",
                                        "properties": {
                                            "amount":  {"type": "number"},                          # 2
                                            "size":    {"type": "string", "enum": ["small", "medium", "large"]}, #medium
                                            "unit":    {"type": "string", "enum": ["piece", "bowl", "glass", "tbsp", "cup", "slice", "plate", "katori", "roti"]},  # katori
                                            "weight":  {"type": "number"},                          # 100
                                            "weight_unit": {"type": "string", "enum": ["g", "ml"]} # g
                                        },
                                        "required": ["amount", "unit", "weight", "weight_unit"]
                                    },
                        "confidence": {"type": "number", "description": "Confidence score for this dish, 0–1"},
                    },
                    "required": ["kcal", "protein", "fat", "carbs", "fiber", "sugar", "quantity", "confidence"],
                },
            },
            "total": {
                "type": "object",
                "description": "Sum of macros across all dishes (the full meal total).",
                "properties": {
                    "kcal":    {"type": "number"},
                    "protein": {"type": "number"},
                    "fat":     {"type": "number"},
                    "carbs":   {"type": "number"},
                    "fiber":   {"type": "number"},
                    "sugar":   {"type": "number"},
                },
                "required": ["kcal", "protein", "fat", "carbs", "fiber", "sugar"],
            },
            "overall_confidence": {
                "type": "number",
                "description": "Overall confidence in the analysis, 0–1.",
            },
            "notes": {
                "type": "string",
                "description": "Optional field containing assumption or uncertainties.",
            },
        },
        "required": ["dishes", "total", "overall_confidence"],
    },
}