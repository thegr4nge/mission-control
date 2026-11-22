# Eval results, 2026-11-23

Ten tasks, scores 0-1 (mean over tasks). Scoring is described in `README.md`; per-task tables are in `results/*.md`.

| Model | Route | Tool use | Correct | Grounded | Explanation | Tool errors |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | Claude Code CLI, Pro plan | 1.00 | 1.00 | 0.98 | 1.00 | 0 |
| Groq `openai/gpt-oss-120b` | free tier | 0.95 | 1.00 | 0.98 | 0.72 | 0 |
| Gemini `gemini-3.1-flash-lite` | free tier | 1.00 | 0.95 | 1.00 | 0.67 | 1 |

Claude's figures combine two runs: on the first, T02 hung once with no output and timed out at 600 s; it
passed in 23 s on its own rerun. Both runs are in `results/`.

## What the misses were

- **Gemini T08:** it called `propagate` without `central_body`, so the tool used the default `sun`.
  It then reported the resulting heliocentric hyperbola (a = 3707 km, e = 1.0) for an Earth satellite.
  The scorer caught it: correctness 0.5.
- **Groq T09:** it worked out the rocket equation itself (4.074 km/s) instead of calling
  `check_delta_v_budget` (4.077 km/s). This passes correctness tolerance but fails tool use (0.5) and
  grounding, which is the hard rule.
- **Claude T05 and T06:** it computed two numbers itself, an altitude from a radius and a difference of
  flight times, and both were flagged as ungrounded.
- **Explanation:** the free models often skip the physics reasons, for example not saying why a 153° orbit
  is retrograde. This rubric is a keyword check, not a judgement of reasoning.

## Not run

- **Gemini flash models:** `gemini-flash-latest`, `gemini-3.8-flash` and `gemini-3.5-flash` returned
  HTTP 503 ("high demand") during the run, so only flash-lite was scored. Rerun with
  `--model gemini --model-name gemini-3.8-flash` when it is available.
- **Ollama:** not installed on this machine.
