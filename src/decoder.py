import json
import re
import numpy as np
from typing import Any, Callable, Optional


TOP_K = 20

NAME_PREFIX = '{"name": "'
PARAMETERS_MARKER = '"parameters": {'

_NUMBER_PREFIX_RE = re.compile(r'^-?(0|[1-9]\d*)?(\.\d*)?([eE][+-]?\d*)?$')


def _current_name_fragment(generated_so_far: str) -> str:
    if generated_so_far.startswith(NAME_PREFIX):
        return generated_so_far[len(NAME_PREFIX):]
    return ""


def _current_parameters_fragment(generated_so_far: str) -> str:
    """Return everything generated so far *inside* the parameters
    object, i.e. what comes after '"parameters": {' (not including
    that opening brace itself). Empty string if we haven't reached
    the parameters object yet."""
    idx = generated_so_far.find(PARAMETERS_MARKER)
    if idx == -1:
        return ""
    return generated_so_far[idx + len(PARAMETERS_MARKER):]


def _name_is_valid_continuation(name_so_far: str, token_text: str, function_names: list[str]) -> bool:
    """Check whether appending `token_text` keeps the function name a
    valid prefix of (or exact match to) one of `function_names`.

    Handles tokens that bundle extra characters after the closing
    quote (e.g. a single token decoding to '", "parameters"') by only
    checking the portion of the combined text up to the first quote.
    """
    combined = name_so_far + token_text
    if '"' in combined:
        name_part = combined.split('"', 1)[0]
        return name_part in function_names
    return any(name.startswith(combined) for name in function_names)


def _is_number_prefix(s: str) -> bool:
    return bool(_NUMBER_PREFIX_RE.match(s))


def _parameters_is_valid_continuation(params_so_far: str, parameter_schema: dict[str, str]) -> bool:
    """Best-effort check that `params_so_far` -- the content generated
    so far *inside* the parameters `{ ... }` object, not including the
    leading '{' -- could still be completed into a JSON object whose
    keys are all present in `parameter_schema` and whose values match
    the declared type ('string', 'number', or 'boolean') for that key.

    Runs a small state machine over the fragment: keys are checked
    against the schema prefix-by-prefix (same trick as function-name
    matching), and once a key's type is known, the value that follows
    is constrained to that type's grammar (quoted text for 'string',
    JSON-number characters for 'number', 'true'/'false' for
    'boolean'). Rejects unknown keys, duplicate keys, and type
    mismatches (e.g. a quoted string for a number field) as soon as
    they appear, rather than only at the end.
    """
    stage = "want_key_or_close"
    key_buf = ""
    used_keys: set[str] = set()
    current_type: Optional[str] = None
    num_buf = ""
    bool_buf = ""
    escape = False

    i = 0
    n = len(params_so_far)
    while i < n:
        ch = params_so_far[i]

        if stage == "want_key_or_close":
            if ch.isspace():
                i += 1; continue
            if ch == '"':
                if not parameter_schema:
                    return False  # no keys are valid at all
                stage = "in_key"; key_buf = ""; escape = False; i += 1; continue
            if ch == '}':
                stage = "done"; i += 1; continue
            return False

        elif stage == "in_key":
            if escape:
                key_buf += ch; escape = False; i += 1; continue
            if ch == "\\":
                escape = True; i += 1; continue
            if ch == '"':
                if key_buf in used_keys or key_buf not in parameter_schema:
                    return False
                stage = "want_colon"; i += 1; continue
            key_buf += ch
            if not any(name.startswith(key_buf) for name in parameter_schema):
                return False
            i += 1; continue

        elif stage == "want_colon":
            if ch.isspace():
                i += 1; continue
            if ch == ':':
                current_type = parameter_schema[key_buf]
                used_keys.add(key_buf)
                stage = "want_value_start"; i += 1; continue
            return False

        elif stage == "want_value_start":
            if ch.isspace():
                i += 1; continue
            if current_type == "string":
                if ch != '"':
                    return False
                stage = "in_string_value"; escape = False; i += 1; continue
            elif current_type == "number":
                if ch not in "-0123456789":
                    return False
                num_buf = ch
                if not _is_number_prefix(num_buf):
                    return False
                stage = "in_number_value"; i += 1; continue
            elif current_type == "boolean":
                if ch not in "tf":
                    return False
                bool_buf = ch
                if not ("true".startswith(bool_buf) or "false".startswith(bool_buf)):
                    return False
                stage = "in_bool_value"; i += 1; continue
            else:
                return False

        elif stage == "in_string_value":
            if escape:
                escape = False; i += 1; continue
            if ch == "\\":
                escape = True; i += 1; continue
            if ch == '"':
                stage = "want_comma_or_close"; i += 1; continue
            i += 1; continue

        elif stage == "in_number_value":
            if ch in "0123456789.eE+-":
                candidate = num_buf + ch
                if not _is_number_prefix(candidate):
                    return False
                num_buf = candidate
                i += 1; continue
            stage = "want_comma_or_close"; continue

        elif stage == "in_bool_value":
            candidate = bool_buf + ch
            if "true".startswith(candidate) or "false".startswith(candidate):
                bool_buf = candidate
                if bool_buf in ("true", "false"):
                    stage = "want_comma_or_close"
                i += 1; continue
            if bool_buf in ("true", "false"):
                stage = "want_comma_or_close"; continue
            return False

        elif stage == "want_comma_or_close":
            if ch.isspace():
                i += 1; continue
            if ch == ',':
                stage = "want_key_or_close"; i += 1; continue
            if ch == '}':
                stage = "done"; i += 1; continue
            return False

        elif stage == "done":
            return False  # nothing valid can follow the closing brace

    return True


def is_valid_json_prefix(text: str) -> bool:
    """Best-effort check that `text` could still be completed into
    valid JSON. Tracks open brackets/quotes (ignoring bracket-like
    characters inside strings) and tries closing them off to see if
    the result parses."""
    if not text.strip():
        return True

    depth_stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth_stack.append(ch)
        elif ch in "}]":
            if not depth_stack:
                return False
            opener = depth_stack.pop()
            if (opener, ch) not in (("{", "}"), ("[", "]")):
                return False

    closers = {"{": "}", "[": "]"}
    candidate = text
    if in_string:
        candidate += '"'
    candidate += "".join(closers[c] for c in reversed(depth_stack))

    try:
        json.loads(candidate)
        return True
    except (json.JSONDecodeError, ValueError):
        return in_string or bool(depth_stack)


def _decode_each(llm_model: Any, token_ids: np.ndarray) -> list[str]:
    """Batch-decode candidate ids, returning each id's own text.
    Falls back to per-id calls if the model's decode doesn't return
    one string per id when given a list."""
    ids = [int(t) for t in token_ids]
    if not ids:
        return []
    decoded = llm_model.decode(ids)
    if isinstance(decoded, list) and len(decoded) == len(ids):
        return decoded
    return [llm_model.decode([i]) for i in ids]


def _select_valid_token(
    logits: np.ndarray,
    llm_model: Any,
    is_valid: Callable[[str], bool],
) -> tuple[Optional[str], Optional[int]]:
    """Sort all token ids by logit once (descending), then walk down
    that order in TOP_K-sized chunks, decoding each chunk in a single
    batched call, until one candidate satisfies `is_valid`. Usually
    stops after the first chunk; only keeps going if the model's best
    guesses don't pan out. Returns (None, None) if nothing qualifies."""
    sorted_ids = np.argsort(logits)[::-1]

    for start in range(0, sorted_ids.shape[0], TOP_K):
        chunk_ids = sorted_ids[start: start + TOP_K]
        texts = _decode_each(llm_model, chunk_ids)
        for token_id, text in zip(chunk_ids, texts):
            if is_valid(text):
                return text, int(token_id)

    return None, None


def select_next_token(logits: list,
                      generated_so_far: str,
                      llm_model: Any,
                      phase: Any,
                      function_names: Optional[list[str]] = None,
                      parameter_schema: Optional[dict[str, str]] = None) -> tuple[str, int]:
    """Pick the highest-scoring token whose text keeps
    `generated_so_far` a valid JSON prefix and, depending on `phase`,
    also satisfies a stricter constraint:
      - during the function-name phase, keeps the function name a
        valid prefix of one of `function_names`.
      - during the parameter phase, keeps the parameters object a
        valid prefix of a JSON object whose keys/value types match
        `parameter_schema` (a dict of param name -> 'string' |
        'number' | 'boolean' for the function that was selected).
    All constraints are checked together on the same candidate pool
    (expanded as needed) so a token can't slip through one check by
    only having been vetted against the other. Falls back to the
    plain argmax token if nothing qualifies, so generation never
    stalls.

    `function_names` is required when `phase.status == "function"`.
    `parameter_schema` is required when `phase.status == "parameter"`."""

    logits = np.asarray(logits, dtype=float)
    name_so_far = _current_name_fragment(generated_so_far)
    params_so_far = _current_parameters_fragment(generated_so_far)
    in_function_phase = phase.status == "function"
    in_parameter_phase = phase.status == "parameter"

    if in_function_phase and not function_names:
        raise ValueError(
            "select_next_token requires a non-empty `function_names` "
            "list while phase.status == 'function'"
        )
    if in_parameter_phase and parameter_schema is None:
        raise ValueError(
            "select_next_token requires a `parameter_schema` dict "
            "while phase.status == 'parameter'"
        )

    def is_valid(text: str) -> bool:
        if not is_valid_json_prefix(generated_so_far + text):
            return False
        if in_function_phase:
            return _name_is_valid_continuation(name_so_far, text, function_names)
        if in_parameter_phase:
            return _parameters_is_valid_continuation(params_so_far + text, parameter_schema)
        return True

    text, token_id = _select_valid_token(logits, llm_model, is_valid)
    if text is not None:
        return text, token_id

    # Nothing in the whole vocab qualified (shouldn't normally happen) -
    # fall back to the plain argmax token so generation never stalls.
    best_id = int(np.argmax(logits))
    best_text = _decode_each(llm_model, np.array([best_id]))[0]
    return best_text, best_id
