"""
Marker detection, segment boundary extraction, and dead-end labeling.

Segment model: within a think block (or the full generated text if no tags),
a segment is the span between two consecutive self-correction markers.
Segments are numbered 0, 1, 2, ... in order of appearance.
"""

import re
import numpy as np

# ── Self-correction markers ───────────────────────────────────────────────────

_MARKER_RE = re.compile(
    r"\b(?:wait,?\s|actually,?\s|that'?s\s+(?:wrong|incorrect|not\s+right)"
    r"|let\s+me\s+(?:reconsider|re-?think|start\s+over|try\s+again)"
    r"|no,?\s+wait\b|hold\s+on\b|wait\s+a\s+(?:minute|moment|second)\b)",
    re.IGNORECASE,
)
_MIN_GAP = 50  # chars; prevents clustering "wait, actually, hmm"


def detect_markers(text):
    """Return list of (char_start, char_end) for self-correction markers."""
    markers = []
    for m in _MARKER_RE.finditer(text):
        if not markers or m.start() - markers[-1][0] >= _MIN_GAP:
            markers.append((m.start(), m.end()))
    return markers


# ── Think block extraction ────────────────────────────────────────────────────

_BOXED_RE       = re.compile(r"\\boxed\{")
_THINK_OPEN_RE  = re.compile(r"<think>",  re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def extract_think_span(text):
    """
    Return (think_text, start, end) — the reasoning portion of generated text.
    Tries explicit <think> tags first, then falls back to last \\boxed paragraph.
    """
    o = _THINK_OPEN_RE.search(text)
    c = _THINK_CLOSE_RE.search(text)
    if o and c:
        return text[o.end():c.start()], o.end(), c.start()

    boxes = list(_BOXED_RE.finditer(text))
    if boxes:
        end = text.rfind("\n\n", 0, boxes[-1].start())
        end = end if end != -1 else boxes[-1].start()
        return text[:end], 0, end

    return text, 0, len(text)


# ── Segment building ──────────────────────────────────────────────────────────

def build_segments(think_text, markers):
    """
    Return list of {seg_id, char_start, char_end, text} dicts.
    markers: output of detect_markers(think_text).
    """
    bounds = [0] + [m[0] for m in markers] + [len(think_text)]
    return [
        {"seg_id": i, "char_start": a, "char_end": b,
         "text": think_text[a:b], "n_chars": b - a}
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
    ]


# ── Dead-end labeling (requires post-hoc attention, see label_attention.py) ──

def label_dead_ends(segments, post_marker_attn_mass, threshold=0.05):
    """
    Given per-segment mean post-marker attention mass (list of floats,
    one per segment), mark segments as dead-end if mass < threshold.
    Returns segments list with 'attention_mass' and 'is_dead_end' added.
    """
    for seg, mass in zip(segments, post_marker_attn_mass):
        seg["attention_mass"] = mass
        seg["is_dead_end"]    = mass < threshold
    return segments


# ── Full pipeline (character-level, no model needed) ─────────────────────────

def segment_trace(trace):
    """
    Segment a raw trace dict. Returns annotation dict or None if unusable.
    Does not require attention signal — dead-end flags left empty until
    label_attention.py runs the post-hoc pass.
    """
    text = trace.get("generated_text", "")
    if not text:
        return None

    think_text, think_start, think_end = extract_think_span(text)
    markers  = detect_markers(think_text)
    segments = build_segments(think_text, markers)

    return {
        "n_markers":        len(markers),
        "n_segments":       len(segments),
        "marker_positions": [m[0] for m in markers],
        "segment_spans":    segments,
        "dead_end_flags":   [],   # populated by label_attention.py
        "dead_end_fraction": 0.0,
        "correct":          trace.get("correct", False),
    }
