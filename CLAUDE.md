# Coding Standards

Write minimal, correct code. Before any line: is this strictly necessary? If not, omit it.
Prefer 50 clear lines over 200 defensive ones.

**Before writing code:** state the approach in plain English and flag assumptions about scope.
**At each step:** "would the task fail without this?" If no, skip it.

No speculative abstractions. No "just in case" error handling. No unasked-for configurability.
No docstrings on obvious functions. No comments unless the logic is non-obvious.
If it wasn't asked for, don't add it.
