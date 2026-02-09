from __future__ import annotations

import time
from datetime import datetime, timezone


def now_ts() -> int:
    return int(time.time())


def ts_to_dt_utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)