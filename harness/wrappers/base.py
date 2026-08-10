"""Base GuardrailWrapper — common interface for all guardrail systems being benchmarked.

Each concrete wrapper implements:
  - load() — instantiate model/tokenizer/client
  - score(text) -> dict — returns {is_unsafe: bool, category: Optional[str], raw: dict}

A guardrail's job here is binary safety classification of an INPUT prompt
(not output classification). Some systems can do both — we evaluate the input task only.
"""
from typing import Optional, Dict, Any, List
import time


class GuardrailWrapper:
    """Common interface."""

    name: str = "base"

    def load(self):
        """Load model/tokenizer. Called once. Idempotent."""
        raise NotImplementedError

    def score(self, text: str) -> Dict[str, Any]:
        """Return verdict for one prompt.

        Returns dict with at least:
          - is_unsafe (bool)
          - category (Optional[str]) — system-specific category label or None
          - raw (dict) — full model response for diagnostics
        """
        raise NotImplementedError

    def score_batch(self, texts: List[str], progress_every: int = 50) -> List[Dict[str, Any]]:
        """Default sequential batch — wrappers can override for true batching."""
        results = []
        t0 = time.time()
        for i, t in enumerate(texts):
            try:
                results.append(self.score(t))
            except Exception as e:
                results.append({"is_unsafe": False, "category": None, "raw": {"error": str(e)}})
            if (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(texts) - i - 1) / rate
                print(f"    [{self.name}] {i+1}/{len(texts)}  rate={rate:.1f}/s  ETA {eta:.0f}s", flush=True)
        return results
