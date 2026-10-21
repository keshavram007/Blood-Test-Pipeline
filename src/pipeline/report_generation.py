import os
import time
from google import genai

from src.pipeline.clinical_context import format_context_for_llm


SYSTEM_PROMPT = """Write a short summary (3-5 sentences) explaining blood test results \
to a patient with NO medical knowledge. Imagine you are a friendly GP explaining results \
to a worried patient who has never studied biology.

BANNED WORDS/PHRASES — do not use ANY of these or similar clinical terms:
"glycaemic", "dysglycaemia", "glycated haemoglobin", "insulin resistance", "systemic",
"metabolic shifts", "anomaly score", "biomarker", "concentrated", "variability",
"susceptibility", "designation", "non-diabetic" (as an adjective)

INSTEAD, use everyday words like: "blood sugar", "your body", "test results",
"a bit higher/lower than normal", "worth checking again"

Example of the WRONG tone (too clinical):
"Elevated fasting glucose with normal HbA1c suggests early impaired fasting glycaemia."

Example of the RIGHT tone (plain English):
"Your blood sugar was a little higher than normal on the day of your test, but your
longer-term blood sugar average looks fine. This can sometimes be an early warning
sign, so it's worth getting checked again."

Rules:
1. Write like you're talking to a friend, not writing a medical report.
2. Do NOT repeat numbers or biomarker values — those are already shown in a table.
3. Do NOT state the risk classification or confidence percentage — describe what it
means in plain words instead.
4. Never say "you have diabetes" — say things like "it would be worth getting this
checked further with your doctor."
5. If critical alerts are present, calmly say the patient should seek medical help soon.
6. Keep it warm, calm, and reassuring — not alarming, not cold and clinical.
7. Maximum 5 short sentences.
"""


def generate_clinical_narrative(context: dict, api_key: str = None, max_retries: int = 3) -> str:
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    context_text = format_context_for_llm(context)

    full_prompt = f"""{SYSTEM_PROMPT}

Here is the structured clinical data for this patient (for your reference —
do not repeat this data, only explain what it means in plain English):

{context_text}

Write the plain-English summary now."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=full_prompt
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"API error ({e}), retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise