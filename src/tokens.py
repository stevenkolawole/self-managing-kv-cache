import re

# Special token strings the model emits inline
FORGET_TOKEN   = lambda n: f"<FORGET seg_{n}>"
BOOKMARK_TOKEN = lambda n: f"<BOOKMARK seg_{n}>"
SUMMARY_OPEN   = "<SUMMARY>"
SUMMARY_CLOSE  = "</SUMMARY>"

# Vocabulary entries to add to the tokenizer (segments 0–63 is enough)
def special_tokens(max_segs=64):
    tokens = [SUMMARY_OPEN, SUMMARY_CLOSE]
    for n in range(max_segs):
        tokens += [FORGET_TOKEN(n), BOOKMARK_TOKEN(n)]
    return tokens

_FORGET_RE   = re.compile(r"<FORGET\s+seg_(\d+)>")
_BOOKMARK_RE = re.compile(r"<BOOKMARK\s+seg_(\d+)>")

def parse_management_tokens(text):
    """Return list of {type, seg, start, end} dicts, sorted by position."""
    events = []
    for m in _FORGET_RE.finditer(text):
        events.append({"type": "forget", "seg": int(m.group(1)),
                       "start": m.start(), "end": m.end()})
    for m in _BOOKMARK_RE.finditer(text):
        events.append({"type": "bookmark", "seg": int(m.group(1)),
                       "start": m.start(), "end": m.end()})
    return sorted(events, key=lambda e: e["start"])
