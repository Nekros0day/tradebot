from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, getcontext

getcontext().prec = 28


def D(x: str | float | int | Decimal) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def round_down_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    # floor to nearest step
    return (value // step) * step


def to_str(d: Decimal) -> str:
    # normalize without scientific notation
    s = format(d, "f")
    # strip trailing zeros
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"