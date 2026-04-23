# stereotype-bench

A measure-agnostic LLM benchmark engine for **stereotype-association measures**.

The engine handles the boring (but important) parts: model adapters, prompt rendering, generation, scoring, cost tracking with a hard budget guard, and result storage. **Stereotype measures** plug in via Python entry points, so a measure (e.g. *Gender Stereotypical Association*) can live in its own package — public, private, or pinned-to-paper-supplement — without forking this repo.

## Status

Early scaffold. The first measure (`gsa-core`) is **embargoed pending publication**; the engine ships with a built-in `stub` measure that returns 0.0 so the full pipeline is exercisable end-to-end without it.

## Architecture

```
                ┌────────────────────────┐
                │ stereotype-bench       │
                │  (engine, public)      │
                │  • providers           │
                │  • tasks               │
                │  • runner              │
                │  • cost tracking       │
                │  • plots               │
                └──────────┬─────────────┘
                           │ entry-point: stereotype_bench.measures
              ┌────────────┼────────────┐
              │            │            │
         ┌────┴────┐  ┌────┴─────┐ ┌────┴────┐
         │  stub   │  │ gsa-core │ │   ...   │
         │(builtin)│  │(private) │ │ (other) │
         └─────────┘  └──────────┘ └─────────┘
```

## Quick start

```sh
git clone https://github.com/jakeazcona/stereotype-bench.git
cd stereotype-bench
uv sync --all-extras
cp .env.example .env  # then edit and add your OPENROUTER_API_KEY
uv run stereotype-bench list-measures
uv run stereotype-bench list-models
uv run pytest
```

Run the example experiment (uses the `stub` measure, so all scores are 0.0 — swap to `gsa` once `gsa-core` is installed):

```sh
uv run stereotype-bench run configs/experiments/gsa_first_impression_v1.yaml
uv run stereotype-bench costs
```

## Cost tracking

Every API call is logged to a SQLite DB at `~/.stereotype_bench/costs.sqlite` (override with `STEREOTYPE_BENCH_DB_PATH`). The runner enforces a per-experiment budget ceiling (`budget_usd` in the experiment YAML) and refuses calls that would exceed it; pass `--force` to override.

```sh
uv run stereotype-bench costs                     # all-time, grouped by model
uv run stereotype-bench costs --run-id <id>       # single run
uv run stereotype-bench costs --csv claim.csv     # CSV (e.g. for reimbursement)
```

OpenRouter returns per-call cost directly when available; otherwise the runner falls back to `configs/pricing.yaml` (versioned in-repo — keep updated as providers change prices).

## Adding a new measure

A measure is any class implementing `stereotype_bench.measures.protocol.StereotypeMeasure`:

```python
class MyMeasure:
    name = "my-measure"
    axis_labels = ("low-axis-label", "high-axis-label")
    def score(self, text: str) -> float: ...
```

In your measure package's `pyproject.toml`:

```toml
[project.entry-points."stereotype_bench.measures"]
my-measure = "my_pkg.measure:MyMeasure"
```

Install it into the same environment (`uv pip install -e ../my-measure-pkg`) and `stereotype-bench list-measures` will pick it up.

## Models

Out of the box the engine talks to **OpenRouter**, which fronts GPT, Claude, Gemini, Llama, Grok, Kimi, and many others under one API key. Direct provider adapters can be added under `src/stereotype_bench/providers/` if you want to bypass OpenRouter's routing markup.

## License

MIT — see [LICENSE](LICENSE).

The `gsa-core` measure (separate repo) is *not* MIT and remains under embargo until publication.
