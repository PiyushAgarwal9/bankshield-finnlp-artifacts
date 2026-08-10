"""OpenAI Moderation API wrapper — `omni-moderation-latest`.

Closed-source baseline. NOT a full guardrail (just multi-label classification),
but commonly used in safety papers as commercial reference.

API: https://platform.openai.com/docs/guides/moderation
Free for OpenAI customers.
"""
import os
import time
from .base import GuardrailWrapper


class OpenAIModerationWrapper(GuardrailWrapper):
    name = "openai_moderation"
    model_id = "omni-moderation-latest"

    def __init__(self, api_key=None, retry_delay=1.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY env var")
        self.client = None
        self.retry_delay = retry_delay

    def load(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        print(f"  OpenAI Moderation API ({self.model_id}) ready", flush=True)

    def score(self, text):
        for attempt in range(3):
            try:
                resp = self.client.moderations.create(model=self.model_id, input=text)
                r = resp.results[0]
                # Get top firing category
                cats = [(name, score) for name, score in r.category_scores.model_dump().items()
                        if r.categories.model_dump().get(name)]
                top = max(cats, key=lambda x: x[1]) if cats else (None, 0)
                return {
                    "is_unsafe": r.flagged,
                    "category": top[0],
                    "raw": {"flagged": r.flagged, "categories": r.categories.model_dump(),
                            "scores": {k: round(v, 4) for k, v in r.category_scores.model_dump().items()}},
                }
            except Exception as e:
                if attempt == 2:
                    return {"is_unsafe": False, "category": None, "raw": {"error": str(e)}}
                time.sleep(self.retry_delay * (attempt + 1))
