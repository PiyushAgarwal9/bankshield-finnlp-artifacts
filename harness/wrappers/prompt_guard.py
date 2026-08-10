"""Prompt Guard 2 (86M) — Meta's tiny injection-only classifier.

Sequence-classification head. Binary: BENIGN (0) vs MALICIOUS (1).
Multilingual (per card). Inputs >512 tokens must be chunked + max-aggregate.

Source: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
"""
import torch
import torch.nn.functional as F
from .base import GuardrailWrapper


class PromptGuard2Wrapper(GuardrailWrapper):
    name = "prompt_guard_2"
    model_id = "meta-llama/Llama-Prompt-Guard-2-86M"

    def __init__(self, device="cuda", threshold=0.5):
        self.device = device
        self.threshold = threshold
        self.tokenizer = None
        self.model = None

    def load(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        print(f"  PromptGuard 2 (86M) loaded on {self.device}", flush=True)

    @torch.no_grad()
    def score(self, text):
        # Chunk to 512 tokens; aggregate by max
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        chunk_size = 510  # leave room for [CLS] [SEP]
        chunks = [tokens[i:i + chunk_size] for i in range(0, max(len(tokens), 1), chunk_size)] or [[]]
        max_p = 0.0
        for chunk in chunks:
            t = self.tokenizer.decode(chunk, skip_special_tokens=True) if chunk else text
            inputs = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
            p = probs[1].item()
            if p > max_p: max_p = p
        return {
            "is_unsafe": max_p >= self.threshold,
            "category": "injection" if max_p >= self.threshold else None,
            "raw": {"p_malicious": max_p, "threshold": self.threshold, "n_chunks": len(chunks)},
        }
