"""Timestamp canonicalisation and legacy-archive upgrade tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from app import service
from app.db import UpgradeError, connect, init_db
from app.timestamps import from_ts, is_canonical_ts, to_ts


@pytest.mark.parametrize("year", [1, 2, 12, 123, 999, 1000, 2026, 9999])
def test_to_ts_pads_year_to_four_digits(year):
    value = datetime(year, 1, 2, 3, 4, 5, 678901, tzinfo=UTC)
    text = to_ts(value)
    assert text == f"{year:04d}-01-02T03:04:05.678901Z"
    assert from_ts(text) == value


def test_to_ts_keeps_microseconds_and_converts_offsets():
    from datetime import timedelta, timezone

    value = datetime(2026, 5, 6, 7, 8, 9, 1234, tzinfo=timezone(timedelta(hours=8)))
    assert to_ts(value) == "2026-05-05T23:08:09.001234Z"


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("0001-01-01T00:00:00.000000Z", datetime(1, 1, 1, tzinfo=UTC)),
        ("1-01-01T00:00:00.000000Z", datetime(1, 1, 1, tzinfo=UTC)),
        ("12-02-03T04:05:06.123456Z", datetime(12, 2, 3, 4, 5, 6, 123456, tzinfo=UTC)),
        ("999-12-31T23:59:59.000001Z", datetime(999, 12, 31, 23, 59, 59, 1, tzinfo=UTC)),
        ("2026-01-02T03:04:05Z", datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)),
        ("2026-01-02T03:04:05.5Z", datetime(2026, 1, 2, 3, 4, 5, 500000, tzinfo=UTC)),
        ("2026-01-02T03:04:05+02:00", datetime(2026, 1, 2, 1, 4, 5, tzinfo=UTC)),
    ],
)
def test_from_ts_accepts_legacy_and_offset_values(stored, expected):
    assert from_ts(stored) == expected


def test_from_ts_rejects_unusable_values():
    for stored in ("", "not-a-time", "2026-13-01T00:00:00Z", "2026-01-01T25:00:00Z"):
        with pytest.raises(ValueError):
            from_ts(stored)


def test_canonical_detection():
    assert is_canonical_ts("0001-01-01T00:00:00.000000Z")
    assert not is_canonical_ts("1-01-01T00:00:00.000000Z")
    assert not is_canonical_ts("2026-01-01T00:00:00Z")


def _legacy_archive(path) -> None:
    """Create an archive as an older release left it (unpadded years)."""
    init_db(path)
    conn = connect(path)
    conn.execute(
        "INSERT INTO ndt_orders (order_no, weld_no, completed_at, start_mm, end_mm) "
        "VALUES ('NDT-OLD', 'W1', '1-01-01T00:00:00.000000Z', 0, 1000)"
    )
    conn.execute(
        "INSERT INTO certificates (cert_no, weld_no, issued_at, start_mm, end_mm) "
        "VALUES ('CERT-OLD', 'W1', '2026-01-10T00:00:00.000000Z', 0, 1000)"
    )
    conn.execute(
        "INSERT INTO certificate_citations (cert_no, ndt_no, position) "
        "VALUES ('CERT-OLD', 'NDT-OLD', 0)"
    )
    conn.execute(
        "INSERT INTO repairs (order_no, weld_no, occurred_at, start_mm, end_mm) "
        "VALUES ('REP-OLD', 'W1', '999-06-01T00:00:00.000000Z', 100, 200)"
    )
    conn.execute(
        "INSERT INTO repairs (order_no, weld_no, occurred_at, start_mm, end_mm) "
        "VALUES ('REP-PLACEHOLDER', 'W1', '1-01-01T00:00:00.000000Z', 300, 400)"
    )
    conn.execute(
        "INSERT INTO repair_ingest_audit (repair_no, action, weld_no, occurred_at, "
        "start_mm, end_mm, candidates_checked, changed_certificates, created_at) "
        "VALUES ('REP-OLD', 'created', 'W1', '999-06-01T00:00:00.000000Z', 100, 200, 1, '[]', "
        "'1000-01-01T00:00:00.000000Z')"
    )
    conn.close()


def test_upgrade_is_idempotent_and_preserves_identifiers(tmp_path):
    db_path = tmp_path / "archive.db"
    _legacy_archive(db_path)

    assert init_db(db_path) == 4
    assert init_db(db_path) == 0  # second run is a no-op

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT order_no FROM ndt_orders").fetchall()[0][0] == "NDT-OLD"
        assert conn.execute("SELECT completed_at FROM ndt_orders").fetchone()[0] == (
            "0001-01-01T00:00:00.000000Z"
        )
        assert conn.execute("SELECT issued_at FROM certificates").fetchone()[0] == (
            "2026-01-10T00:00:00.000000Z"
        )
        rows = conn.execute(
            "SELECT order_no, occurred_at FROM repairs ORDER BY order_no"
        ).fetchall()
        assert [(r["order_no"], r["occurred_at"]) for r in rows] == [
            ("REP-OLD", "0999-06-01T00:00:00.000000Z"),
            ("REP-PLACEHOLDER", "0001-01-01T00:00:00.000000Z"),
        ]
        audit = conn.execute(
            "SELECT repair_no, action, occurred_at, created_at FROM repair_ingest_audit"
        )
        assert [tuple(r) for r in audit] == [
            (
                "REP-OLD",
                "created",
                "0999-06-01T00:00:00.000000Z",
                "1000-01-01T00:00:00.000000Z",
            )
        ]
        citations = conn.execute("SELECT cert_no, ndt_no FROM certificate_citations").fetchall()
        assert [tuple(r) for r in citations] == [("CERT-OLD", "NDT-OLD")]
    finally:
        conn.close()


def test_gap_report_works_on_legacy_archive_before_conversion(tmp_path):
    """The rules use real instants even if a legacy value escaped conversion."""
    db_path = tmp_path / "archive.db"
    _legacy_archive(db_path)
    conn = connect(db_path)
    try:
        report = service.check_gap(conn, "CERT-OLD")
        # REP-OLD happened in year 999, between film (year 1) and issuance.
        assert [(g.start_mm, g.end_mm) for g in report.gaps] == [(100, 200)]
        # The placeholder repair at the film instant does not cut anything.
        assert len(report.evidence[0].invalidating_repairs) == 1
    finally:
        conn.close()


def test_interrupted_conversion_can_be_retried_on_the_same_volume(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import app.db as db_module

    db_path = tmp_path / "archive.db"
    _legacy_archive(db_path)

    # One row per transaction so the simulated crash lands between batches.
    monkeypatch.setattr(db_module, "_CONVERSION_BATCH", 1)
    real_write_tx = db_module.write_tx
    calls = {"n": 0}

    @contextmanager
    def flaky_write_tx(conn):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("simulated crash mid-upgrade")
        with real_write_tx(conn):
            yield conn

    monkeypatch.setattr(db_module, "write_tx", flaky_write_tx)
    with pytest.raises(sqlite3.OperationalError):
        db_module.upgrade_legacy_timestamps(db_module.connect(db_path))

    conn = db_module.connect(db_path)
    assert conn.execute("SELECT completed_at FROM ndt_orders").fetchone()[0] == (
        "0001-01-01T00:00:00.000000Z"
    )
    assert conn.execute("SELECT occurred_at FROM repairs ORDER BY order_no").fetchone()[0] == (
        "999-06-01T00:00:00.000000Z"
    )
    conn.close()

    monkeypatch.setattr(db_module, "write_tx", real_write_tx)
    # Only the rows not committed by the interrupted run remain to convert.
    assert init_db(db_path) == 3
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT order_no, occurred_at FROM repairs ORDER BY order_no"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("REP-OLD", "0999-06-01T00:00:00.000000Z"),
            ("REP-PLACEHOLDER", "0001-01-01T00:00:00.000000Z"),
        ]
    finally:
        conn.close()


def test_api_lists_normalize_legacy_timestamps_before_conversion(tmp_path):
    db_path = tmp_path / "archive.db"
    _legacy_archive(db_path)
    conn = connect(db_path)
    try:
        repairs = {r.order_no: r.occurred_at for r in service.list_repairs(conn)}
        assert repairs["REP-OLD"] == "0999-06-01T00:00:00.000000Z"
        assert repairs["REP-PLACEHOLDER"] == "0001-01-01T00:00:00.000000Z"
        certs = {c.cert_no: c.issued_at for c in service.list_certificates(conn)}
        assert certs["CERT-OLD"] == "2026-01-10T00:00:00.000000Z"
    finally:
        conn.close()


def test_upgrade_paginates_across_batches(tmp_path, monkeypatch):
    import app.db as db_module

    db_path = tmp_path / "archive.db"
    init_db(db_path)
    conn = connect(db_path)
    for i in range(5):
        conn.execute(
            "INSERT INTO repairs (order_no, weld_no, occurred_at, start_mm, end_mm) "
            "VALUES (?, 'W1', ?, 0, 10)",
            (f"R{i}", f"{i + 1}-01-01T00:00:00.000000Z"),
        )
    conn.close()

    monkeypatch.setattr(db_module, "_CONVERSION_BATCH", 2)
    assert init_db(db_path) == 5
    conn = connect(db_path)
    try:
        values = [r[0] for r in conn.execute("SELECT occurred_at FROM repairs ORDER BY order_no")]
        assert values == [
            "0001-01-01T00:00:00.000000Z",
            "0002-01-01T00:00:00.000000Z",
            "0003-01-01T00:00:00.000000Z",
            "0004-01-01T00:00:00.000000Z",
            "0005-01-01T00:00:00.000000Z",
        ]
    finally:
        conn.close()


def test_unusable_timestamp_aborts_upgrade_without_partial_row(tmp_path):
    db_path = tmp_path / "archive.db"
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO repairs (order_no, weld_no, occurred_at, start_mm, end_mm) "
        "VALUES ('R-BAD', 'W1', 'garbage', 0, 10)"
    )
    conn.close()
    with pytest.raises(UpgradeError):
        init_db(db_path)
