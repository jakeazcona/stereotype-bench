"""Score VLM predictions in `~/projects/vlm-preds/` with the GSA measure.

Reads each of 6 JSONL files (3 Qwen2.5-VL variants × 2 prompt conditions),
parses model/variant/gender from filename + CFD image code, builds the text
to score per Jake's rules, batch-embeds via the HF Inference Endpoint, and
writes augmented JSONL records to `runs/vlm/` carrying both the original
fields and engine-schema fields (model, text, score, metadata.{gender,variant}).

The output schema matches what `plots/means_ci.py:load_results` consumes,
so the scored files can be passed straight to the comparison plot.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from gsa_core.embeddings import HuggingFaceInferenceEmbedder
from gsa_core.fit import score_embeddings
from gsa_core.measure import GSAMeasure
from stereotype_bench.text_clean import clean_model_output

VLM_DIR = Path.home() / "projects" / "vlm-preds"
OUT_DIR = Path.home() / "projects" / "stereotype-bench" / "runs" / "vlm"

# Filename → (model_id, variant) mapping.
FILE_MAP: dict[str, tuple[str, str]] = {
    "preds_base_describe30.jsonl":                    ("qwen-2.5-vl/untuned",     "plain"),
    "preds_base_three_traits_open.jsonl":             ("qwen-2.5-vl/untuned",     "three_traits"),
    "preds_ft_baseline_describe30.jsonl":             ("qwen-2.5-vl/ft-baseline", "plain"),
    "preds_ft_baseline_three_traits_open.jsonl":      ("qwen-2.5-vl/ft-baseline", "three_traits"),
    "preds_ft_stereo_describe30.jsonl":               ("qwen-2.5-vl/ft-stereo",   "plain"),
    "preds_ft_stereo_three_traits_open.jsonl":        ("qwen-2.5-vl/ft-stereo",   "three_traits"),
}

# Captures the trait label after a "1." / "2." / "3." line marker, tolerant of
# **bold** wrappers and trailing ":" explanations. Captures everything up to a
# colon, asterisk, or newline.
_TRAIT_LINE_RE = re.compile(
    r"^\s*\d+[.)]\s*(?:\*\*)?\s*([^:*\n]+?)\s*(?:\*\*)?\s*(?::|$)",
    re.MULTILINE,
)


def parse_gender(image_filename: str) -> str | None:
    """CFD code: `CFD-XY-...` where Y in {M, F}. Returns 'male' / 'female' / None."""
    parts = image_filename.split("-", 2)
    if len(parts) < 2 or len(parts[1]) != 2:
        return None
    code = parts[1][1].upper()
    if code == "M":
        return "male"
    if code == "F":
        return "female"
    return None


def extract_three_traits(generation: str) -> list[str] | None:
    """Return up to 3 trait labels from a numbered-list generation, else None.

    Returns None if fewer than 3 numbered traits can be extracted (caller falls
    back to scoring raw cleaned text).
    """
    matches = _TRAIT_LINE_RE.findall(generation)
    cleaned = [m.strip().strip("*").strip() for m in matches]
    cleaned = [c for c in cleaned if c]
    if len(cleaned) < 3:
        return None
    return cleaned[:3]


def build_scoring_text(
    *, model_id: str, variant: str, generation: str
) -> tuple[str, str]:
    """Return (text_to_score, scoring_mode).

    scoring_mode is one of:
      - "cleaned":  clean_model_output(generation)
      - "parsed":   "trait1, trait2, trait3" (ft three_traits, parsed)
    """
    cleaned = clean_model_output(generation)
    if variant != "three_traits":
        return cleaned, "cleaned"
    # untuned three_traits → raw cleaned (per Jake's decision)
    if model_id.endswith("/untuned"):
        return cleaned, "cleaned"
    # ft models three_traits → try to parse 3 traits, comma-joined, no Oxford
    traits = extract_three_traits(generation)
    if traits is None:
        return cleaned, "cleaned"
    return ", ".join(traits), "parsed"


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_one_file(
    *,
    src_path: Path,
    out_path: Path,
    model_id: str,
    variant: str,
    measure: GSAMeasure,
    batch_size: int = 16,
) -> dict:
    """Score one VLM JSONL → write augmented JSONL with score + engine fields."""
    records = load_records(src_path)
    n = len(records)
    print(f"  [{src_path.name}] {n} records → model={model_id}, variant={variant}")

    # Build per-record (text_to_score, scoring_mode, gender) up front.
    texts: list[str] = []
    modes: list[str] = []
    genders: list[str | None] = []
    for rec in records:
        text, mode = build_scoring_text(
            model_id=model_id, variant=variant, generation=rec.get("generation", "")
        )
        texts.append(text)
        modes.append(mode)
        genders.append(parse_gender(rec.get("image", "")))

    # Ensure fit is loaded (uses bundled .npz cache; no refit).
    fit = measure._ensure_fit()
    backend = measure.embedding_backend

    # Batch-embed everything; project against the fit.
    scores: list[float] = []
    t_start = time.time()
    n_batches = (n + batch_size - 1) // batch_size
    for bi in range(n_batches):
        chunk = texts[bi * batch_size : (bi + 1) * batch_size]
        embs = backend.embed(chunk)
        chunk_scores = score_embeddings(embs, fit)
        scores.extend(float(s) for s in chunk_scores)
        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t_start
            done = (bi + 1) * batch_size
            rate = done / elapsed if elapsed else 0
            print(f"    batch {bi + 1}/{n_batches} ({min(done, n)}/{n}) "
                  f"elapsed={elapsed:.1f}s, {rate:.1f} rec/s", flush=True)

    parsed_count = sum(1 for m in modes if m == "parsed")
    print(f"    parsed-traits scoring: {parsed_count}/{n} rows "
          f"(fallback raw: {n - parsed_count})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rec, text, mode, gender, score in zip(records, texts, modes, genders, scores):
            out_rec = {
                **rec,                                  # preserve original fields
                "model": model_id,
                "text": text,
                "score": score,
                "scoring_mode": mode,
                "metadata": {
                    "gender": gender,
                    "variant": variant,
                    "image": rec.get("image"),
                    "scoring_mode": mode,
                    **(rec.get("meta") or {}),
                },
            }
            fh.write(json.dumps(out_rec) + "\n")

    return {
        "n": n,
        "parsed": parsed_count,
        "n_male": sum(1 for g in genders if g == "male"),
        "n_female": sum(1 for g in genders if g == "female"),
        "n_gender_unknown": sum(1 for g in genders if g is None),
    }


def main() -> int:
    if not VLM_DIR.is_dir():
        print(f"missing {VLM_DIR}", file=sys.stderr)
        return 1

    measure = GSAMeasure()
    # Force a warm-up of the fit (loads .npz cache once; no per-file repeat).
    measure._ensure_fit()
    print(f"GSA fit loaded: model={measure._fit.model_name}, "
          f"femtyp_range=[{measure._fit.femtyp_min}, {measure._fit.femtyp_max}]")

    summary = {}
    t0 = time.time()
    for fname, (model_id, variant) in FILE_MAP.items():
        src = VLM_DIR / fname
        out = OUT_DIR / (src.stem + ".scored.jsonl")
        summary[fname] = score_one_file(
            src_path=src,
            out_path=out,
            model_id=model_id,
            variant=variant,
            measure=measure,
        )

    print(f"\nDone in {time.time() - t0:.1f}s")
    for f, s in summary.items():
        print(f"  {f}: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
