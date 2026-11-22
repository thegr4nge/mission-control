# Mission eval

Ten mission tasks with known answers (`tasks.py`), each scored on four things (`scoring.py`):

| Score | What it measures |
|---|---|
| tool_use | fraction of the required tools called successfully (call errors counted separately) |
| correctness | fraction of answer keys within tolerance of the published answer |
| grounding | fraction of numbers in the reply that trace to a tool result or the prompt |
| explanation | keyword rubric: did the explanation mention the concepts that matter. Crude, not a judgement of reasoning |

Answers come from NASA/TM-2010-216764, Curtis, Vallado and the Mariner 10 record; see each
`Expect.source` and `physics/references.py` for how each was checked.

## Running

```
pixi run eval --model claude                  # Claude Code CLI on your own plan; no API key, no API spend
pixi run eval --model gemini                  # Google AI Studio free tier
pixi run eval --model groq                    # Groq free tier
pixi run eval --model ollama                  # local Ollama
pixi run eval --model groq --tasks T01,T05    # a subset
pixi run eval --model gemini --list-models    # what your key can call
pixi run eval --rescore evals/results/<file>.json
```

Gemini and Groq read their keys from `.env` in the repo root (gitignored, never committed):

```
GEMINI_API_KEY=...      # https://aistudio.google.com/apikey
GROQ_API_KEY=...        # https://console.groq.com/keys
GEMINI_MODEL=...        # optional overrides of the defaults in agents/model_adapter.py
GROQ_MODEL=...
OLLAMA_MODEL=...
```

The Claude run loads no user settings, hooks, plugins or CLAUDE.md, and has only the physics tools.
Results, with full transcripts, land in `evals/results/`.
