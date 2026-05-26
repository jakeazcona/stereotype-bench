"""Tail-friendly progress logger for long experiment runs.

Emits periodic one-line status snapshots plus per-model and error breakdowns to
a standard Python logger, so `tail -f run.log` on a nohup'd process shows live
progress without relying on TTY escape codes.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field

_log = logging.getLogger("stereotype_bench.progress")


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class _ModelStats:
    ok: int = 0
    fail: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ProgressTracker:
    """Single-consumer tracker; expects updates from the main thread only.

    Use `update_success` / `update_failure` from the `as_completed` loop. Call
    `maybe_snapshot()` after each update; it self-throttles.
    """

    n_total: int
    n_already_done: int = 0  # for resumed runs, shown in snapshot
    snapshot_every_n: int = 100
    snapshot_every_s: float = 20.0
    rate_window_s: float = 60.0
    model_breakdown_every: int = 5  # every Nth snapshot, include per-model detail

    start_time: float = field(default_factory=time.monotonic)
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    error_counts: Counter = field(default_factory=Counter)
    per_model: dict = field(default_factory=dict)
    _recent: deque = field(default_factory=deque)  # (monotonic_ts, completed)
    _last_snapshot_time: float = 0.0
    _last_snapshot_completed: int = 0
    _snapshot_n: int = 0

    def _stats(self, model: str) -> _ModelStats:
        s = self.per_model.get(model)
        if s is None:
            s = _ModelStats()
            self.per_model[model] = s
        return s

    def update_success(
        self,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float,
    ) -> None:
        s = self._stats(model)
        s.ok += 1
        s.input_tokens += int(input_tokens or 0)
        s.output_tokens += int(output_tokens or 0)
        s.cost_usd += float(cost_usd or 0.0)
        self.completed += 1
        self.succeeded += 1
        self._tick()

    def update_failure(self, model: str, error_type: str) -> None:
        s = self._stats(model)
        s.fail += 1
        self.completed += 1
        self.failed += 1
        self.error_counts[error_type] += 1
        self._tick()

    def _tick(self) -> None:
        now = time.monotonic()
        self._recent.append((now, self.completed))
        # Drop samples older than rate_window_s
        cutoff = now - self.rate_window_s
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def _rate(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        t0, n0 = self._recent[0]
        t1, n1 = self._recent[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return (n1 - n0) / dt

    def maybe_snapshot(self, force: bool = False) -> None:
        now = time.monotonic()
        since_last_t = now - self._last_snapshot_time
        since_last_n = self.completed - self._last_snapshot_completed
        if not force and since_last_t < self.snapshot_every_s \
                and since_last_n < self.snapshot_every_n:
            return
        self._emit_snapshot(now)
        self._last_snapshot_time = now
        self._last_snapshot_completed = self.completed
        self._snapshot_n += 1

    def _emit_snapshot(self, now: float) -> None:
        done_this_run = self.completed
        done_total = self.n_already_done + done_this_run
        frac = done_total / max(1, self.n_total)
        elapsed = now - self.start_time
        rate = self._rate()
        remaining = max(0, self.n_total - done_total)
        eta = (remaining / rate) if rate > 0 else float("inf")
        total_cost = sum(s.cost_usd for s in self.per_model.values())

        bar = _fmt_bar(frac)
        _log.info(
            "%s %d/%d %.1f%% rate=%.1f/s elapsed=%s eta=%s spend=$%.4f ok=%d fail=%d",
            bar,
            done_total,
            self.n_total,
            frac * 100.0,
            rate,
            _fmt_duration(elapsed),
            _fmt_duration(eta) if rate > 0 else "?",
            total_cost,
            self.succeeded,
            self.failed,
        )

        # Per-model detail every Nth snapshot (keeps per-tick log quiet).
        if self._snapshot_n % max(1, self.model_breakdown_every) == 0:
            for model in sorted(self.per_model):
                s = self.per_model[model]
                attempts = s.ok + s.fail
                fail_pct = (100.0 * s.fail / attempts) if attempts else 0.0
                _log.info(
                    "  %-42s ok=%d fail=%d (%.1f%%) in=%d out=%d cost=$%.4f",
                    model,
                    s.ok,
                    s.fail,
                    fail_pct,
                    s.input_tokens,
                    s.output_tokens,
                    s.cost_usd,
                )

        if self.error_counts:
            top = ", ".join(
                f"{k}={v}" for k, v in self.error_counts.most_common(5)
            )
            _log.info("  errors: %s", top)

    def final_summary(self) -> None:
        """Flush a final snapshot and a full per-model + error breakdown."""
        self.maybe_snapshot(force=True)
        elapsed = time.monotonic() - self.start_time
        total_cost = sum(s.cost_usd for s in self.per_model.values())
        _log.info(
            "=== run complete: trials=%d ok=%d fail=%d elapsed=%s cost=$%.4f ===",
            self.completed, self.succeeded, self.failed,
            _fmt_duration(elapsed), total_cost,
        )
        for model in sorted(self.per_model):
            s = self.per_model[model]
            attempts = s.ok + s.fail
            fail_pct = (100.0 * s.fail / attempts) if attempts else 0.0
            _log.info(
                "  %-42s ok=%d fail=%d (%.1f%%) in=%d out=%d cost=$%.4f",
                model, s.ok, s.fail, fail_pct,
                s.input_tokens, s.output_tokens, s.cost_usd,
            )
        if self.error_counts:
            _log.info("  error types (all):")
            for k, v in self.error_counts.most_common():
                _log.info("    %6d  %s", v, k)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up a clean line-based format on the root logger's stderr handler.

    Idempotent: safe to call multiple times (won't stack handlers).
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Replace existing handlers rather than append, so repeat calls stay clean.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()  # stderr by default
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    # httpx is chatty at INFO about every single request; quiet it.
    logging.getLogger("httpx").setLevel(logging.WARNING)
