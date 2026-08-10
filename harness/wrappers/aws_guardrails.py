"""AWS Bedrock Guardrails wrapper — the v3 'fair config' guardrail
(all filters HIGH, no denied topics — same configuration as the June 2026
full benchmark, per its protocol).

Credentials + guardrail identity come from .env.aws in the repo root
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, GUARDRAIL_ID,
GUARDRAIL_VERSION).

BENCH_DIRECTION=output maps to ApplyGuardrail source='OUTPUT' (native).
"""
import os
import time
from pathlib import Path

from .base import GuardrailWrapper

ENV_FILE = Path(__file__).parent.parent.parent / ".env.aws"


def _load_env():
    vals = {}
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                vals[k] = v
    return vals


class AWSGuardrailsWrapper(GuardrailWrapper):
    name = "aws_guardrails"

    def __init__(self):
        self.client = None
        self.cfg = None

    def load(self):
        import boto3
        self.cfg = _load_env()
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.cfg["AWS_REGION"],
            aws_access_key_id=self.cfg["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=self.cfg["AWS_SECRET_ACCESS_KEY"],
        )
        print(f"  AWS Guardrails client ready (guardrail {self.cfg['GUARDRAIL_ID']} "
              f"v{self.cfg['GUARDRAIL_VERSION']}, {self.cfg['AWS_REGION']})", flush=True)

    def _call(self, text, source):
        return self.client.apply_guardrail(
            guardrailIdentifier=self.cfg["GUARDRAIL_ID"],
            guardrailVersion=self.cfg["GUARDRAIL_VERSION"],
            source=source,
            content=[{"text": {"text": text[:10000]}}],
        )

    def score(self, text):
        source = "OUTPUT" if os.environ.get("BENCH_DIRECTION", "input") == "output" else "INPUT"
        try:
            resp = self._call(text, source)
        except Exception as e:
            if "ThrottlingException" in str(e):
                time.sleep(5)
                try:
                    resp = self._call(text, source)
                except Exception as e2:
                    return {"is_unsafe": False, "category": None, "raw": {"error": str(e2)[:200]}}
            else:
                return {"is_unsafe": False, "category": None, "raw": {"error": str(e)[:200]}}
        action = resp.get("action", "NONE")
        assessments = resp.get("assessments", [])
        cat = None
        for a in assessments:
            for f in (a.get("contentPolicy", {}) or {}).get("filters", []):
                if f.get("action") == "BLOCKED":
                    cat = f.get("type")
                    break
        time.sleep(0.1)  # gentle rate limit
        return {"is_unsafe": action == "GUARDRAIL_INTERVENED",
                "category": cat,
                "raw": {"action": action, "source": source}}
