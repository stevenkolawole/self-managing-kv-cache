"""Shared utilities: model loading, answer extraction, trace I/O."""

import json
import os
import re
import sys
from pathlib import Path

HF_CACHE = os.environ.get("HF_HOME", "/data/hf_cache/skolawol")

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text):
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def answers_match(pred, gt):
    if pred is None:
        return False
    def norm(s):
        s = re.sub(r"\\(?:dfrac|left|right|text\{[^}]*\})", "", s)
        return s.lower().replace(" ", "").replace(",", "")
    if norm(pred) == norm(gt):
        return True
    return sorted(norm(p) for p in pred.split(",")) == sorted(norm(g) for g in gt.split(","))


def as_legacy_kv(past_kv):
    """Normalise past_key_values to list of (key, value) per layer."""
    if hasattr(past_kv, "layers"):
        return [(l.keys, l.values) for l in past_kv.layers]
    if hasattr(past_kv, "key_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    return [(l[0], l[1]) for l in past_kv]


def load_model(name, eager=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    aliases = {"7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
               "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
    repo = aliases.get(name, name)
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype="auto", device_map="auto", cache_dir=HF_CACHE)
    if eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs).eval()
    return model, tok


def load_traces(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_trace(trace, path):
    with open(path, "a") as f:
        f.write(json.dumps(trace) + "\n")
