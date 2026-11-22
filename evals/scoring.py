"""Score one model transcript against a task.

Four scores, each 0-1:
  tool_use     required tools called successfully (fraction), shown with the call error count
  correctness  answer keys within tolerance of the known answer (fraction)
  grounding    numbers in the final answer and explanation that trace to a tool result or the prompt
  explanation  crude rubric: required concepts mentioned (fraction), zero if no explanation at all

The explanation score is a keyword check, not a judgement of reasoning quality.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from evals.tasks import Task

SYSTEM_PROMPT = (
    "You are the navigator on a spacecraft mission design team. You have physics tools computed with ESA's pykep. "
    "Every number you state about an orbit, trajectory, date or delta-v must come from a tool result: never "
    "estimate or calculate one yourself. Call the tools you need, then explain in under 150 words what you "
    "computed and why the answer is what it is."
)


def answer_instructions(task: Task) -> str:
    keys = ", ".join(task.expected)
    return (
        f"{task.prompt}\n\nFinish your reply with one line that starts with ANSWER: followed by a JSON object with "
        f"exactly these keys: {keys}. Dates as YYYY-MM-DD strings, numbers as plain JSON numbers in the units the key "
        "names, booleans as true or false, sequences as lists of lowercase planet names."
    )


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result_text: str


@dataclass
class Transcript:
    task_id: str
    model: str
    final_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None  # harness or API failure; the task scores zero
    seconds: float = 0.0
    turns: int = 0


@dataclass
class Score:
    task_id: str
    model: str
    tool_use: float
    tool_errors: int
    correctness: float
    grounding: float
    explanation: float
    answer: dict[str, Any] | None
    key_results: dict[str, bool]
    ungrounded_numbers: list[str]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_answer(text: str) -> dict[str, Any] | None:
    """The JSON object after the last ANSWER: marker, tolerating markdown fences and trailing text."""
    idx = text.rfind("ANSWER:")
    if idx < 0:
        return None
    tail = text[idx + len("ANSWER:"):].strip().strip("`").strip()
    start = tail.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(tail[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                obj = json.loads(tail[start:i + 1])
            except json.JSONDecodeError:
                return None
            return obj if isinstance(obj, dict) else None
    return None


def _to_date(v: Any) -> date | None:
    try:
        return date.fromisoformat(str(v).strip()[:10])
    except ValueError:
        return None


def _type_label(v: Any) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(v).lower()).removeprefix("type")
    return {"1": "i", "2": "ii"}.get(s, s)


def check_key(expect, got: Any) -> bool:
    if got is None:
        return False
    k = expect.kind
    if k == "number" or k == "max":
        if isinstance(got, bool):
            return False
        try:
            x = float(got)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(x):
            return False
        if k == "max":
            return 0.0 <= x <= expect.value
        return abs(x - expect.value) <= expect.tol * abs(expect.value)
    if k == "date":
        d, ref = _to_date(got), _to_date(expect.value)
        return d is not None and abs((d - ref).days) <= expect.tol
    if k == "text":
        return _type_label(got) == _type_label(expect.value)
    if k == "bool":
        return isinstance(got, bool) and got is expect.value
    if k == "sequence":
        return isinstance(got, list) and [str(s).strip().lower() for s in got] == expect.value
    raise ValueError(f"unknown expectation kind {k}")


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})(?::\d{2})?Z?)?\b")
# Citations are not physics numbers: "Example 4.3", "Table 2", "Eq. 5.28".
CITATION_RE = re.compile(r"\b(?:example|table|eq\.?|equation|section|figure|fig\.?|chapter)\s+[\d.\-]+", re.I)

# Geometry constants a correct explanation may quote without a tool (transfer-type boundary, full turn).
GEOMETRIC_CONSTANTS = (90.0, 180.0, 270.0, 360.0)
# Tool inputs that are the model's own choices rather than physics results (search bounds, orbit
# altitudes, durations, margins). Numbers passed in any other argument must already be grounded.
FREE_ARGUMENTS = frozenset({"min_time_of_flight_days", "max_time_of_flight_days", "duration_days", "duration_s",
                            "departure_orbit_altitude_km", "capture_orbit_altitude_km", "margin_percent",
                            "max_flybys"})
# A tool gives hours, the model may quote minutes; days and seconds; km and m; fractions and percent.
UNIT_FACTORS = (1.0, 24.0, 1 / 24.0, 60.0, 1 / 60.0, 3600.0, 1 / 3600.0, 86400.0, 1 / 86400.0,
                1000.0, 1e-3, 100.0, 0.01)


_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺",
                              "0123456789-+")


def _normalise(text: str) -> str:
    """Models write the minus sign as U+2212 and digits grouped with commas."""
    text = text.replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    # 1.3x10^-9, 1.3 × 10⁻⁹ and the like become 1.3e-9.
    text = text.translate(_SUPERSCRIPTS)
    return re.sub(r"(\d(?:\.\d+)?)\s*[x×⋅*]\s*10\s*\^?\s*\(?([-+]?\d+)\)?", r"\1e\2", text)


def _numbers(text: str) -> list[float]:
    """Magnitudes of every number in text, with ISO dates split into year, month, day, hour, minute."""
    text = _normalise(text)
    out = [float(g) for m in DATE_RE.finditer(text) for g in m.groups() if g]
    out += [abs(float(m)) for m in NUMBER_RE.findall(DATE_RE.sub(" ", text))]
    return out


def _arg_numbers(args: Any, free: bool = False) -> tuple[list[float], list[float]]:
    """(free, constrained) numeric values in a tool's arguments."""
    free_vals, bound_vals = [], []
    if isinstance(args, dict):
        for k, v in args.items():
            f, b = _arg_numbers(v, free or k in FREE_ARGUMENTS)
            free_vals += f
            bound_vals += b
    elif isinstance(args, list):
        for v in args:
            f, b = _arg_numbers(v, free)
            free_vals += f
            bound_vals += b
    elif isinstance(args, (int, float)) and not isinstance(args, bool):
        (free_vals if free else bound_vals).append(abs(float(args)))
    return free_vals, bound_vals


def _is_rounding(x_text: str, y: float) -> bool:
    """x as written is y rounded to the digits shown (half a unit in x's last place)."""
    x = abs(float(x_text))
    mant = re.sub(r"[eE].*", "", x_text.lstrip("-"))
    exp = int(re.search(r"[eE]([-+]?\d+)", x_text).group(1)) if re.search(r"[eE]", x_text) else 0
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    half = 0.5 * 10.0 ** (exp - decimals)
    return abs(x - y) <= half * (1 + 1e-9) + 1e-12


def _needs_trace(x_text: str) -> bool:
    """Every non-integer, and every integer of 10 or more ('2 burns' and '4 s.f.' are not physics)."""
    return "." in x_text or "e" in x_text.lower() or abs(float(x_text)) >= 10


def grounding(final_text: str, tool_texts: list[str], prompt: str,
              tool_args: list[dict] | None = None) -> tuple[float, list[str]]:
    """Fraction of the reply's numbers that are roundings of a source number.

    Sources: the prompt, tool results, and the free tool arguments (search bounds and the like).
    tool_texts[i] is the result of the call made with tool_args[i] (successful calls only, in order).
    A call whose constrained inputs are not themselves grounded counts as laundering: its result is
    not a source, and each invented input is reported. An empty reply scores 0.
    """
    if not final_text.strip():
        return 0.0, []
    tool_args = tool_args if tool_args is not None else [{} for _ in tool_texts]
    base = _numbers(prompt)
    results: list[float] = []
    bad: list[str] = []
    for text, args in zip(tool_texts, tool_args):
        free, constrained = _arg_numbers(args)
        known = base + results + free
        known_scaled = [y * f for y in known for f in UNIT_FACTORS]
        # An input is grounded if it is a rounding of a known number, or a known number is a rounding of it
        # (the model passed full precision where the tool printed 4 s.f.).
        invented = [v for v in constrained
                    if not any(_is_rounding(repr(v), y) or _is_rounding(repr(y), v) for y in known_scaled)]
        base += free
        if invented:
            bad += [f"input:{v:g}" for v in invented]
            continue
        results += _numbers(text)
    pool = [y * f for y in base + results for f in UNIT_FACTORS] + list(GEOMETRIC_CONSTANTS)
    text = DATE_RE.sub(" ", CITATION_RE.sub(" ", _normalise(final_text)))
    candidates = [m for m in NUMBER_RE.findall(text) if _needs_trace(m)]
    bad += [m for m in candidates if not any(_is_rounding(m, y) for y in pool)]
    total = len(candidates) + sum(b.startswith("input:") for b in bad)
    if total == 0:
        return 1.0, []
    return 1.0 - len(bad) / total, bad


def explanation_score(task: Task, final_text: str) -> float:
    prose = final_text[: final_text.rfind("ANSWER:")] if "ANSWER:" in final_text else final_text
    if len(prose.split()) < 15:
        return 0.0
    if not task.concepts:
        return 1.0
    low = prose.lower().replace("-", " ")
    hit = sum(any(alt.lower().replace("-", " ") in low for alt in alts) for alts in task.concepts)
    return hit / len(task.concepts)


def score(task: Task, tr: Transcript) -> Score:
    if tr.error:
        return Score(task.id, tr.model, 0.0, 0, 0.0, 0.0, 0.0, None, {k: False for k in task.expected}, [], tr.error)
    ok_tools = {c.name for c in tr.tool_calls if c.ok}
    tool_use = sum(t in ok_tools for t in task.required_tools) / len(task.required_tools)
    errors = sum(not c.ok for c in tr.tool_calls)
    answer = parse_answer(tr.final_text)
    keys = {k: check_key(e, (answer or {}).get(k)) for k, e in task.expected.items()}
    correctness = sum(keys.values()) / len(keys)
    ok_calls = [c for c in tr.tool_calls if c.ok]
    g, bad = grounding(tr.final_text, [c.result_text for c in ok_calls], answer_instructions(task),
                       [c.arguments for c in ok_calls])
    return Score(task.id, tr.model, tool_use, errors, correctness, g, explanation_score(task, tr.final_text),
                 answer, keys, bad, None)
