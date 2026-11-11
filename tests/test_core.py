"""Tests for config, event logic and the Excel log.  Run:  python -m pytest"""

import json
from datetime import date, datetime, timedelta

import pytest
from openpyxl import load_workbook

from hrbt import error_log
from hrbt.config import Config, ConfigError, load_config, normalise_hhmm
from hrbt.event_tracker import EventTracker, in_window
from hrbt.excel_logger import (HEADERS, ExcelLogger, create_workbook,
                               read_blocked_seconds_between)
from hrbt.stream_monitor import moving_flags


@pytest.fixture(autouse=True)
def _tmp_error_log(tmp_path):
    error_log.set_log_path(tmp_path / "errors.log")


# ------------------------------------------------------------------ config
def test_config_created_with_defaults(tmp_path):
    p = tmp_path / "config.json"
    cfg, created = load_config(p)
    assert created and p.exists()
    data = json.loads(p.read_text())
    assert data["window_start"] == "0600" and data["window_end"] == "2000"
    assert data["event_threshold_seconds"] == 30
    assert cfg.streams_are_placeholders()
    cfg2, created2 = load_config(p)
    assert not created2 and cfg2 == cfg


@pytest.mark.parametrize("raw,expected", [("0600", "0600"), ("600", "0600"), (600, "0600"),
                                          ("06:00", "0600"), ("2000", "2000"), ("2400", "2400")])
def test_hhmm_ok(raw, expected):
    assert normalise_hhmm(raw) == expected


@pytest.mark.parametrize("raw", ["2500", "0660", "abc", "", "2401", "12345"])
def test_hhmm_bad(raw):
    with pytest.raises(ConfigError):
        normalise_hhmm(raw)


def test_bad_threshold():
    with pytest.raises(ConfigError):
        Config(event_threshold_seconds="x").validate()
    with pytest.raises(ConfigError):
        Config(event_threshold_seconds=0).validate()


# ------------------------------------------------------------ time window
def t(h, m=0, s=0, day=13):
    return datetime(2026, 10, day, h, m, s)


def test_in_window():
    assert in_window(t(6), 360, 1200)
    assert in_window(t(19, 59), 360, 1200)
    assert not in_window(t(20), 360, 1200)
    assert not in_window(t(5, 59), 360, 1200)
    # overnight window 2200 -> 0600
    assert in_window(t(23), 1320, 360) and in_window(t(2), 1320, 360)
    assert not in_window(t(12), 1320, 360)
    assert in_window(t(3), 0, 0)  # equal = all day


# ---------------------------------------------------------- event tracker
def feed(tracker, start, pattern, step=0.5):
    """pattern: list of (seconds, value). Returns the time after feeding."""
    now = start
    for duration, value in pattern:
        for _ in range(int(duration / step)):
            tracker.observe(now, value)
            now += timedelta(seconds=step)
    return now


def test_event_counted_once_and_timed():
    tr = EventTracker(5, 360, 1200)
    feed(tr, t(10), [(3, True), (20, False), (3, True)])
    out = tr.drain(flush=True)
    events, secs = out[date(2026, 10, 13)]
    assert events == 1
    # 20 s empty -> ~20 s blocked (threshold seconds included)
    assert 19 <= secs <= 20


def test_short_gaps_are_not_events():
    tr = EventTracker(5, 360, 1200)
    feed(tr, t(10), [(4, False), (1, True)] * 10)
    assert tr.drain(flush=True) == {}


def test_two_events():
    tr = EventTracker(5, 360, 1200)
    feed(tr, t(10), [(10, False), (2, True), (8, False), (1, True)])
    events, _ = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 2


def test_outside_window_ignored_and_event_ends_at_window_close():
    tr = EventTracker(5, 360, 1200)
    feed(tr, t(5, 50), [(60, False)])          # before 0600
    assert tr.drain(flush=True) == {}
    tr = EventTracker(5, 360, 1200)
    feed(tr, t(19, 59, 0), [(120, False)])     # runs across 2000
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and 59 <= secs <= 60   # empty from 19:59:00, counted until 20:00


def test_outage_not_counted_and_does_not_trigger():
    tr = EventTracker(5, 360, 1200)
    now = feed(tr, t(10), [(3, False), (60, None), (3, False), (1, True)])
    assert tr.drain(flush=True) == {}
    # event in progress continues through an outage without being double counted
    tr = EventTracker(5, 360, 1200)
    feed(tr, now, [(10, False), (30, None), (10, False), (1, True)])
    events, secs = tr.drain(flush=True)[now.date()]
    assert events == 1 and 19 <= secs <= 20   # ~10 before outage + ~10 after, outage excluded


def test_midnight_split():
    tr = EventTracker(5, 0, 0)  # all day
    feed(tr, t(23, 59, 0), [(120, False)])
    out = tr.drain(flush=True)
    assert out[date(2026, 10, 13)][0] == 1
    assert 59 <= out[date(2026, 10, 13)][1] <= 60   # empty from 23:59:00
    assert out[date(2026, 10, 14)] == (0, 60) or 59 <= out[date(2026, 10, 14)][1] <= 60


def test_drain_carries_fractions():
    tr = EventTracker(1, 0, 0)
    feed(tr, t(10), [(3.5, False)])
    first = tr.drain()
    rest = tr.drain(flush=True)
    total = sum(v[1] for v in first.values()) + sum(v[1] for v in rest.values())
    assert total in (2, 3)


# ------------------------------------------------------------------ motion
def test_moving_flags():
    a = (100, 100, 200, 160)
    still = (101, 100, 201, 160)
    moved = (140, 100, 240, 160)
    assert moving_flags([a], None) == [True]
    assert moving_flags([still], [[a], [a], [a], [a]]) == [False]
    assert moving_flags([moved], [[a], [a], [a], [a]]) == [True]
    # matches 2 s back (next car in line) but not 0.5-1.5 s back -> moving
    assert moving_flags([still], [[moved], [moved], [moved], [a]]) == [True]
    # small box changing size by 2 px: same place
    assert moving_flags([(10, 10, 32, 31)], [[(10, 10, 30, 29)]] * 4) == [False]
    # small box wobbling 2 px: same place (old overlap test called this moving)
    small, wobble = (10, 10, 30, 30), (12, 8, 31, 29)
    assert moving_flags([wobble], [[small]] * 4) == [False]
    # box growing ~30 % (car driving at the camera): moving
    grown = (100 - 15, 100 - 10, 200 + 15, 160 + 10)
    assert moving_flags([grown], [[a]] * 4) == [True]


# ------------------------------------------------------------------- excel
def test_excel_create_add_reopen(tmp_path):
    p = tmp_path / "log.xlsx"
    create_workbook(p)
    lg = ExcelLogger(p)
    lg.open()
    d = date(2026, 10, 13)
    assert lg.add(1, {d: (1, 30)}) is True
    lg.add(2, {d: (0, 12)})
    lg.add(1, {d: (2, 5), d + timedelta(days=1): (1, 1)})
    assert lg.totals(d) == (3, 35, 0, 12)
    assert lg.close()

    ws = load_workbook(p).active
    assert [c.value for c in ws[1]] == HEADERS
    assert ws.cell(2, 1).value.date() == d
    assert [ws.cell(2, c).value for c in range(2, 6)] == [3, 35, 0, 12]
    assert ws.max_row == 3

    lg2 = ExcelLogger(p)   # reopening continues the same rows
    lg2.open()
    lg2.add(2, {d: (1, 4)})
    assert lg2.totals(d) == (3, 35, 1, 16)
    assert lg2.close()


def test_three_minute_blockage_with_15s_threshold():
    # normal 10 s gaps are ignored; a 3-minute blockage is one event of ~3 min
    tr = EventTracker(15, 360, 1200)
    now = feed(tr, t(12), [(10, False), (1, True)] * 5)
    assert tr.drain(flush=True) == {}
    feed(tr, now, [(180, False), (1, True)])
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and 179 <= secs <= 180


def test_road_empty_before_window_opens_only_counts_from_window():
    # empty since 05:50; window opens 06:00 -> event at 06:00 credits only the
    # 15 s threshold, not the 10 minutes before the window
    tr = EventTracker(15, 360, 1200)
    feed(tr, t(5, 50), [(11 * 60, False)])
    events, secs = tr.drain(flush=True)[date(2026, 10, 13)]
    assert events == 1 and 70 <= secs <= 76    # 15 s credit + ~60 s after 06:00



def test_last_30_days_totals(tmp_path):
    p = tmp_path / "log.xlsx"
    create_workbook(p)
    lg = ExcelLogger(p)
    lg.open()
    today = date(2026, 10, 13)
    lg.add(1, {today: (1, 600), today - timedelta(days=29): (1, 3000),   # inside
               today - timedelta(days=30): (1, 99999)})                  # day 31: outside
    lg.add(2, {today - timedelta(days=5): (2, 4500)})
    first = today - timedelta(days=29)
    assert lg.blocked_seconds_between(first, today) == (3600, 4500)
    assert lg.close()
    # same answer straight from the file (used while logging is stopped)
    assert read_blocked_seconds_between(p, first, today) == (3600, 4500)
    assert read_blocked_seconds_between(tmp_path / "missing.xlsx", first, today) == (0, 0)
