"""SQLite access helpers.

The archive is a single SQLite file.  Connections run in autocommit mode so
that every write endpoint can take an explicit ``BEGIN IMMEDIATE`` transaction
and commit or roll back as one unit.

Startup also upgrades older volumes in place.  Timestamps written by releases
without four-digit year padding (e.g. ``1-01-01T00:00:00.000000Z``) are
rewritten to the canonical form, one small transaction at a time.  Each batch
is atomic, so a conversion that is interrupted leaves every row either
untouched or already canonical and the same volume can be retried.  Only
timestamp columns are touched; order numbers, citations and audit rows are
preserved verbatim.  The service finishes this conversion before accepting
requests, so it never serves a half-converted archive.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .timestamps import from_ts, is_canonical_ts, to_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS ndt_orders (
    order_no      TEXT PRIMARY KEY,
    weld_no       TEXT NOT NULL,
    completed_at  TEXT NOT NULL,
    start_mm      INTEGER NOT NULL,
    end_mm        INTEGER NOT NULL,
    CHECK (end_mm > start_mm)
);

CREATE TABLE IF NOT EXISTS certificates (
    cert_no    TEXT PRIMARY KEY,
    weld_no    TEXT NOT NULL,
    issued_at  TEXT NOT NULL,
    start_mm   INTEGER NOT NULL,
    end_mm     INTEGER NOT NULL,
    CHECK (end_mm > start_mm)
);

CREATE TABLE IF NOT EXISTS certificate_citations (
    cert_no  TEXT NOT NULL REFERENCES certificates(cert_no) ON DELETE CASCADE,
    ndt_no   TEXT NOT NULL REFERENCES ndt_orders(order_no),
    position INTEGER NOT NULL,
    PRIMARY KEY (cert_no, ndt_no)
);

CREATE TABLE IF NOT EXISTS repairs (
    order_no     TEXT PRIMARY KEY,
    weld_no      TEXT NOT NULL,
    occurred_at  TEXT NOT NULL,
    start_mm     INTEGER NOT NULL,
    end_mm       INTEGER NOT NULL,
    CHECK (end_mm > start_mm)
);

CREATE TABLE IF NOT EXISTS repair_ingest_audit (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    repair_no            TEXT NOT NULL,
    action               TEXT NOT NULL,
    weld_no              TEXT NOT NULL,
    occurred_at          TEXT NOT NULL,
    start_mm             INTEGER NOT NULL,
    end_mm               INTEGER NOT NULL,
    candidates_checked   INTEGER NOT NULL,
    changed_certificates TEXT NOT NULL,
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ndt_weld_time
    ON ndt_orders (weld_no, completed_at);
CREATE INDEX IF NOT EXISTS idx_cert_weld_time
    ON certificates (weld_no, issued_at);
CREATE INDEX IF NOT EXISTS idx_repair_weld_time
    ON repairs (weld_no, occurred_at);
CREATE INDEX IF NOT EXISTS idx_citation_ndt
    ON certificate_citations (ndt_no);
"""

# (table, primary key columns, timestamp columns)
_TIMESTAMP_COLUMNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("ndt_orders", ("order_no",), ("completed_at",)),
    ("certificates", ("cert_no",), ("issued_at",)),
    ("repairs", ("order_no",), ("occurred_at",)),
    ("repair_ingest_audit", ("id",), ("occurred_at", "created_at")),
)

_CONVERSION_BATCH = 500


class UpgradeError(RuntimeError):
    """An old archive could not be upgraded safely."""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def upgrade_legacy_timestamps(conn: sqlite3.Connection) -> int:
    """Rewrite non-canonical timestamps into canonical UTC form.

    Returns the number of rows converted.  Idempotent: canonical rows are left
    alone, so an interrupted run can simply be repeated.  Rows are processed in
    small ``BEGIN IMMEDIATE`` batches; a crash between batches leaves each
    committed batch canonical and the rest untouched.  Updates compare the old
    value as well, so a concurrent writer's fresh value is never overwritten
    with a stale conversion.
    """

    converted = 0
    for table, key_columns, ts_columns in _TIMESTAMP_COLUMNS:
        key_sql = ", ".join(key_columns)
        col_sql = ", ".join(ts_columns)
        last_rowid = 0
        while True:
            rows = conn.execute(
                f"SELECT rowid AS _rowid, {key_sql}, {col_sql} FROM {table} "
                "WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, _CONVERSION_BATCH),
            ).fetchall()
            if not rows:
                break
            last_rowid = rows[-1]["_rowid"]
            pending: list[tuple[dict[str, str], dict[str, str], tuple[object, ...]]] = []
            for row in rows:
                updates: dict[str, str] = {}
                originals: dict[str, str] = {}
                for column in ts_columns:
                    stored = row[column]
                    if is_canonical_ts(stored):
                        continue
                    try:
                        updates[column] = to_ts(from_ts(stored))
                    except ValueError as exc:
                        raise UpgradeError(
                            f"{table}.{column} value {stored!r} for key "
                            f"{tuple(row[k] for k in key_columns)!r} is not a usable "
                            "timestamp; refusing to upgrade the archive"
                        ) from exc
                    originals[column] = stored
                if updates:
                    pending.append((updates, originals, tuple(row[k] for k in key_columns)))
            if not pending:
                continue
            with write_tx(conn):
                for updates, originals, key in pending:
                    assignments = ", ".join(f"{column} = ?" for column in updates)
                    guards = " AND ".join(f"{column} = ?" for column in originals)
                    where = " AND ".join(f"{column} = ?" for column in key_columns)
                    conn.execute(
                        f"UPDATE {table} SET {assignments} WHERE {where} AND {guards}",
                        (*updates.values(), *key, *originals.values()),
                    )
            converted += len(pending)
    return converted


def init_db(db_path: str | Path) -> int:
    """Create or upgrade the archive and return the number of converted rows."""
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        return upgrade_legacy_timestamps(conn)
    finally:
        conn.close()


@contextmanager
def write_tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One atomic write unit.  Any exception rolls the whole unit back."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextmanager
def read_tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Stable snapshot for multi-statement reports."""
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
