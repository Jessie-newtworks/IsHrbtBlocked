"""Per-stream stoppage ("event") logic.

Rules
-----
* The detector reports an observation for every video frame: either
  "a vehicle is passing" (True), "no vehicle is passing" (False), or
  "no usable picture" (None - stream down / frozen).
* When no vehicle has been seen for `threshold` consecutive seconds *and* the
  current time is inside the logging window, an event starts: the day's event
  counter goes up by 1 and the blocked-time timer starts running. Blocked
  time is measured from when the road went empty, so it includes the
  threshold seconds (a 3-minute blockage is logged as ~3 minutes).
* The timer keeps running until a vehicle is seen again (or the logging window
  closes). It then pauses until the next event.
* While the picture is unavailable nothing is timed - an outage is never
  counted as blocked time, and the threshold has to be re-observed afterwards.
  An event that was already running before a short outage simply continues
  (it is not counted twice).
* Blocked seconds that cross midnight are split between the two dates.

The tracker never touches Excel. The GUI periodically calls `drain()` to
collect what changed and hands that to the Excel logger.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta


def in_window(now: datetime, start_min: int, end_min: int) -> bool:
    """True if `now` falls inside [start, end). Handles windows past midnight.

    start == end means "all day".
    """
    m = now.hour * 60 + now.minute
    if start_min == end_min:
        return True
    if start_min < end_min:
        return start_min <= m < end_min
    return m >= start_min or m < end_min  # e.g. 2200 -> 0600


@dataclass
class TrackerSnapshot:
    feed_ok: bool
    in_window: bool
    in_event: bool
    clear_seconds: float          # how long the road has been empty
    current_event_seconds: float  # length of the running event (0 if none)


class EventTracker:
    def __init__(self, threshold_seconds: float, window_start_min: int, window_end_min: int):
        self.threshold = float(threshold_seconds)
        self.window_start = window_start_min
        self.window_end = window_end_min
        self._lock = threading.Lock()

        self._clear_since: datetime | None = None
        self._last_tick: datetime | None = None
        self._last_now: datetime | None = None
        self._in_event = False
        self._feed_ok = False
        self._current_event_seconds = 0.0

        self._pending_events: dict[date, int] = defaultdict(int)
        self._pending_seconds: dict[date, float] = defaultdict(float)

    # ---------------------------------------------------------------- updates
    def observe(self, now: datetime, vehicle_passing: bool | None) -> None:
        with self._lock:
            self._last_now = now

            if vehicle_passing is None:            # no usable picture
                self._feed_ok = False
                self._clear_since = None
                self._last_tick = None
                return
            self._feed_ok = True

            if vehicle_passing:                     # traffic is moving
                self._in_event = False
                self._current_event_seconds = 0.0
                self._clear_since = None
                self._last_tick = now
                return

            # ---- no vehicle passing ----
            if self._clear_since is None:
                self._clear_since = now
            window_open = in_window(now, self.window_start, self.window_end)

            if self._in_event:
                if not window_open:
                    self._in_event = False
                    self._current_event_seconds = 0.0
                elif self._last_tick is not None and now > self._last_tick:
                    self._add_seconds(self._last_tick, now)
            elif window_open and (now - self._clear_since).total_seconds() >= self.threshold:
                self._in_event = True
                self._current_event_seconds = 0.0
                self._pending_events[now.date()] += 1
                # Blocked time runs from when the road went empty, so the
                # threshold seconds count too. (If the road was already empty
                # when the window opened, only the threshold itself is credited.)
                empty_for = (now - self._clear_since).total_seconds()
                credit = empty_for if empty_for < self.threshold + 2 else self.threshold
                self._add_seconds(now - timedelta(seconds=credit), now)

            self._last_tick = now

    def _add_seconds(self, start: datetime, end: datetime) -> None:
        """Add blocked time [start, end), splitting it at midnight if needed."""
        self._current_event_seconds += (end - start).total_seconds()
        cursor = start
        while cursor.date() < end.date():
            midnight = datetime.combine(cursor.date() + timedelta(days=1), datetime.min.time(),
                                        tzinfo=cursor.tzinfo)
            self._pending_seconds[cursor.date()] += (midnight - cursor).total_seconds()
            cursor = midnight
        self._pending_seconds[end.date()] += (end - cursor).total_seconds()

    def end_event(self) -> None:
        """Stop any running event (used when monitoring is stopped)."""
        with self._lock:
            self._in_event = False
            self._current_event_seconds = 0.0
            self._clear_since = None
            self._last_tick = None
            self._feed_ok = False

    # ---------------------------------------------------------------- reading
    def drain(self, flush: bool = False) -> dict[date, tuple[int, int]]:
        """Return and clear the changes since the last drain.

        Result: {date: (new_events, new_whole_seconds)}. Fractions of a second
        are carried over to the next drain, unless flush=True (then rounded).
        """
        out: dict[date, tuple[int, int]] = {}
        with self._lock:
            for d in set(self._pending_events) | set(self._pending_seconds):
                events = self._pending_events.pop(d, 0)
                secs = self._pending_seconds.get(d, 0.0)
                whole = int(round(secs)) if flush else int(secs)
                remainder = 0.0 if flush else secs - whole
                if remainder > 1e-9:
                    self._pending_seconds[d] = remainder
                else:
                    self._pending_seconds.pop(d, None)
                if events or whole:
                    out[d] = (events, whole)
        return out

    def pending(self, d: date) -> tuple[int, float]:
        """Not-yet-drained (events, seconds) for date d - for live display."""
        with self._lock:
            return self._pending_events.get(d, 0), self._pending_seconds.get(d, 0.0)

    def snapshot(self, now: datetime | None = None) -> TrackerSnapshot:
        with self._lock:
            now = now or self._last_now or datetime.now()
            clear = (now - self._clear_since).total_seconds() if self._clear_since else 0.0
            return TrackerSnapshot(
                feed_ok=self._feed_ok,
                in_window=in_window(now, self.window_start, self.window_end),
                in_event=self._in_event,
                clear_seconds=max(0.0, clear),
                current_event_seconds=self._current_event_seconds,
            )
