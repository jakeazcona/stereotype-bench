"""SQLite-backed call-cost log."""
from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    purpose TEXT NOT NULL,
    prompt_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);
CREATE INDEX IF NOT EXISTS idx_calls_model ON calls(model);
"""


def default_db_path() -> Path:
    env = os.environ.get("STEREOTYPE_BENCH_DB_PATH")
    if env:
        return Path(env)
    return Path.home() / ".stereotype_bench" / "costs.sqlite"


@dataclass
class CallRecord:
    run_id: str
    provider: str
    model: str
    purpose: str
    prompt_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float
    metadata_json: str | None = None


class CostDB:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, call: CallRecord) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO calls(
                    ts, run_id, provider, model, purpose, prompt_id,
                    input_tokens, output_tokens, cost_usd, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    call.run_id,
                    call.provider,
                    call.model,
                    call.purpose,
                    call.prompt_id,
                    call.input_tokens,
                    call.output_tokens,
                    call.cost_usd,
                    call.metadata_json,
                ),
            )
            return cur.lastrowid or 0

    def total(self, run_id: str | None = None) -> float:
        with self._conn() as c:
            if run_id:
                row = c.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM calls"
                ).fetchone()
            return float(row[0])

    def by_model(self, run_id: str | None = None) -> list[dict]:
        with self._conn() as c:
            if run_id:
                rows = c.execute(
                    """SELECT model, COUNT(*) AS n, SUM(cost_usd) AS cost
                       FROM calls WHERE run_id = ?
                       GROUP BY model ORDER BY cost DESC""",
                    (run_id,),
                ).fetchall()
            else:
                rows = c.execute(
                    """SELECT model, COUNT(*) AS n, SUM(cost_usd) AS cost
                       FROM calls
                       GROUP BY model ORDER BY cost DESC"""
                ).fetchall()
            return [dict(r) for r in rows]

    def to_csv(self, out_path: Path | str, run_id: str | None = None) -> int:
        with self._conn() as c:
            if run_id:
                rows = c.execute(
                    "SELECT * FROM calls WHERE run_id = ? ORDER BY ts",
                    (run_id,),
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM calls ORDER BY ts").fetchall()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            if not rows:
                return 0
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))
        return len(rows)
