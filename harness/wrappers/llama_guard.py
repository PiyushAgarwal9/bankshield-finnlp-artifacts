"""Llama Guard 4 wrapper — Meta's 12B native-multimodal guardrail.

Requires transformers>=4.51.3 (LlamaGuard-preview branch) + hf_xet.
14 hazard categories S1-S14 (MLCommons taxonomy).
Multilingual: en, fr, de, hi, it, pt, es, th. NOT Tamil/Telugu/Kannada.

Source: https://huggingface.co/meta-llama/Llama-Guard-4-12B
"""
import os

import torch
from .base import GuardrailWrapper

LLAMAGUARD4_CATS = {
    "S1": "Violent Crimes", "S2": "Non-Violent Crimes", "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation", "S5": "Defamation", "S6": "Specialized Advice",
    "S7": "Privacy", "S8": "Intellectual Property", "S9": "Indiscriminate Weapons",
    "S10": "Hate", "S11": "Suicide & Self-Harm", "S12": "Sexual Content",
    "S13": "Elections", "S14": "Code Interpreter Abuse",
}


class LlamaGuard4Wrapper(GuardrailWrapper):
    name = "llama_guard_4"
    model_id = "meta-llama/Llama-Guard-4-12B"

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoProcessor, Llama4ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_id, device_map=self.device, torch_dtype=self.dtype,
        )
        self.model.eval()
        # Resolve pad_token_id robustly (avoids "'<' not supported between int and NoneType")
        tok = getattr(self.processor, "tokenizer", None) or self.processor
        self._pad_token_id = (
            getattr(tok, "pad_token_id", None)
            or getattr(tok, "eos_token_id", None)
            or getattr(self.model.config, "eos_token_id", None)
            or 0
        )
        if isinstance(self._pad_token_id, list):
            self._pad_token_id = self._pad_token_id[0]

        # transformers 5.7 has a bug where SlidingWindowLayer fails if sliding_window=None.
        # Patch all config layers to ensure non-None.
        def _patch_cfg(cfg):
            if cfg is None: return
            for attr in ("sliding_window", "max_position_embeddings", "attention_chunk_size"):
                v = getattr(cfg, attr, None)
                if v is None:
                    setattr(cfg, attr, 8192)
        _patch_cfg(self.model.config)
        for sub in ("text_config", "vision_config"):
            _patch_cfg(getattr(self.model.config, sub, None))

        print(f"  Llama Guard 4 (12B) loaded on {self.device} (pad={self._pad_token_id})", flush=True)

    @torch.no_grad()
    def score(self, text):
        if os.environ.get("BENCH_DIRECTION", "input") == "output":
            # Llama Guard classifies the LAST turn — put text as the assistant
            # response after a fixed neutral user turn.
            from .wildguard import NEUTRAL_USER
            messages = [
                {"role": "user", "content": [{"type": "text", "text": NEUTRAL_USER}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=self._pad_token_id,
            cache_implementation="dynamic",  # avoid StaticCache None-len bug
        )
        decoded = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )[0]
        return self._parse(decoded)

    @staticmethod
    def _parse(text):
        text = text.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return {"is_unsafe": False, "category": None, "raw": {"output": text}}
        is_unsafe = lines[0].lower().startswith("unsafe")
        cats = []
        if is_unsafe and len(lines) > 1:
            for l in lines[1:]:
                cats.extend([c.strip() for c in l.split(",") if c.strip().startswith("S")])
        return {"is_unsafe": is_unsafe,
                "category": cats[0] if cats else None,
                "raw": {"output": text, "all_categories": cats,
                        "category_names": [LLAMAGUARD4_CATS.get(c, c) for c in cats]}}
