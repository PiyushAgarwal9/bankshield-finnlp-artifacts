# BankShield wrapper - calls the deployed BankShield HTTP service (distinct from L3Cube-IndicGuard).
"""BankShield wrapper — calls our live HTTP service.

Default: v3 production at localhost:8000 (all 6 binary gates).
Override via BANKSHIELD_ENDPOINT env var to hit v4 ensemble at :8004/v1/scan-v4.

Concurrent score_batch: BANKSHIELD_CONCURRENCY env var (default 16).

Optional HTTP basic auth via the BANKSHIELD_PASS_FILE env var (none by default).
"""
import os, json, base64
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import GuardrailWrapper


class BankShieldWrapper(GuardrailWrapper):
    name = "bankshield"

    def __init__(self, endpoint=None,
                 user="bankshield", pass_file=None):
        self.endpoint = endpoint or os.environ.get(
            "BANKSHIELD_ENDPOINT", "http://localhost:8000/v1/scan"
        )
        self.user = user
        try:
            with open(pass_file) as f:
                pw = f.read().strip()
        except FileNotFoundError:
            pw = ""  # service runs without HTTP basic auth by default
        self.auth = base64.b64encode(f"{user}:{pw}".encode()).decode()

    def load(self):
        # Health check: v3 has /v1/health, v4 has /health
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(self.endpoint)
        is_v4 = "/v1/scan-v4" in self.endpoint
        health_path = "/health" if is_v4 else "/v1/health"
        health_url = urlunsplit((parts.scheme, parts.netloc, health_path, "", ""))
        req = urllib.request.Request(health_url, headers={"Authorization": f"Basic {self.auth}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            print(f"  BankShield live at {self.endpoint}: {r.read().decode()[:80]}", flush=True)

    def _do_request(self, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.endpoint, data=data, method="POST",
                                      headers={"Content-Type": "application/json",
                                               "Authorization": f"Basic {self.auth}"})
        timeout = int(os.environ.get("BANKSHIELD_TIMEOUT", "180"))
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _parse_response(self, resp):
        """
        Parse IG response into binary is_unsafe verdict.

        Two sources, controlled by BANKSHIELD_VERDICT_MODE env var:
          - "action" (default): use final_action / action — what production user sees
                                (post SI-veto, post-lex-veto, post-decision-matrix)
          - "gate":              ANY of 6 binary gates triggered → unsafe
                                 PLUS lexical filter triggered → unsafe
                                 (peer-comparable — matches what peer systems output)
        """
        mode = os.environ.get("BANKSHIELD_VERDICT_MODE", "action")
        # Current the service schema: decision BLOCK|ROUTE|ALLOW (+ flagged bool).
        # Older v3 schema: action/final_action. Support both.
        decision = resp.get("decision")
        if decision is not None:
            action = decision
            if decision == "ALLOW" and resp.get("flagged"):
                action = "FLAG"  # detected-but-allowed still counts as detection
        else:
            action = resp.get("action") or resp.get("final_action") or "allow"
        safe_actions = {"allow", "ALLOW", "ALLOW_REDIRECT"}

        gate_triggered = False
        category_from_gate = None
        gates = resp.get("gate_results") or {}
        for gname, gdata in gates.items():
            if isinstance(gdata, dict) and gdata.get("triggered"):
                gate_triggered = True
                category_from_gate = gname
                break

        # Lexical filter triggered? (in v3 issues array; v4 has lex_veto_fired flag)
        lex_triggered = bool(resp.get("lex_veto_fired"))
        if not lex_triggered:
            for issue in resp.get("issues") or []:
                if isinstance(issue, dict) and issue.get("module") == "lexical_filter":
                    lex_triggered = True
                    break

        if mode == "gate":
            is_unsafe = gate_triggered or lex_triggered
        else:
            is_unsafe = action not in safe_actions

        return {
            "is_unsafe": is_unsafe,
            "category": resp.get("fired_category") or resp.get("danger_type")
                        or category_from_gate or (resp.get("lora") or {}).get("category"),
            "raw": {"action": action,
                    "decision_rationale": (resp.get("decision_rationale") or "")[:200],
                    "gate_triggered": gate_triggered,
                    "lex_triggered": lex_triggered,
                    "verdict_mode": mode,
                    "danger_score": resp.get("danger_score"),
                    "intent_score": resp.get("intent_score"),
                    "session_attack": (resp.get("session") or {}).get("session_attack_score"),
                    "decision_reason": (resp.get("decision_reason") or resp.get("reason", ""))[:200]},
        }

    def score(self, text):
        """Single-turn scoring (compat path).

        BENCH_DIRECTION=output scans text as an assistant-side message
        (service's /v1/scan `direction` field)."""
        is_v4 = "/v1/scan-v4" in self.endpoint
        body = {"turns": [{"role": "user", "text": text}]} if is_v4 else {"text": text}
        direction = os.environ.get("BENCH_DIRECTION", "input")
        if not is_v4 and direction != "input":
            body["direction"] = direction
        try:
            return self._parse_response(self._do_request(body))
        except Exception as e:
            return {"is_unsafe": False, "category": None, "raw": {"error": str(e)}}

    def score_item(self, item):
        """Score a full dataset item (with metadata). For multi-turn datasets,
        uses turns array from metadata to invoke /v1/scan-v4 with full conversation."""
        is_v4 = "/v1/scan-v4" in self.endpoint
        meta = item.get("metadata") or {}
        turns = meta.get("turns")  # multi-turn datasets attach this

        if is_v4 and turns:
            # Multi-turn path: pass full conversation
            body = {"turns": turns}
        elif is_v4:
            body = {"turns": [{"role": "user", "text": item["prompt"]}]}
        else:
            # v3 endpoint — concat prompt only (v3 has no multi-turn API)
            body = {"text": item["prompt"]}
            direction = os.environ.get("BENCH_DIRECTION", "input")
            if direction != "input":
                body["direction"] = direction

        try:
            return self._parse_response(self._do_request(body))
        except Exception as e:
            return {"is_unsafe": False, "category": None, "raw": {"error": str(e)}}

    def score_batch(self, texts, progress_every=50):
        return self._batch([{"prompt": t, "metadata": {}} for t in texts],
                           progress_every, scorer=self.score_item)

    def score_batch_items(self, items, progress_every=50):
        return self._batch(items, progress_every, scorer=self.score_item)

    def _batch(self, items, progress_every, scorer):
        n = len(items)
        results = [None] * n
        concurrency = int(os.environ.get("BANKSHIELD_CONCURRENCY", "16"))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_idx = {ex.submit(scorer, it): i for i, it in enumerate(items)}
            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = {"is_unsafe": False, "category": None, "raw": {"error": str(e)}}
                done += 1
                if done % progress_every == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (n - done) / max(rate, 0.1)
                    print(f"    [{self.name}] {done}/{n}  rate={rate:.1f}/s  ETA {eta:.0f}s", flush=True)
        return results
