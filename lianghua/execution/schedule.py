"""交易时段判断（第十三轮·定时调仓辅助）。"""
from __future__ import annotations

import datetime as _dt

TRADING_AM = (_dt.time(9, 30), _dt.time(11, 30))
TRADING_PM = (_dt.time(13, 0), _dt.time(15, 0))


def is_trading_session(now=None):
    """返回 (bool, str)。周一~周五 09:30-11:30 / 13:00-15:00 为交易中。"""
    now = now or _dt.datetime.now()
    if now.weekday() >= 5:
        return False, "周末休市"
    t = now.time()
    if TRADING_AM[0] <= t <= TRADING_AM[1]:
        return True, "盘中(上午)"
    if TRADING_PM[0] <= t <= TRADING_PM[1]:
        return True, "盘中(下午)"
    if t < TRADING_AM[0]:
        return False, "盘前"
    if TRADING_AM[1] < t < TRADING_PM[0]:
        return False, "午间休市"
    return False, "盘后"


def next_session(now=None):
    """返回下一个交易时段开始时间（datetime）。"""
    now = now or _dt.datetime.now()
    for (start, _) in (TRADING_AM, TRADING_PM):
        dt = now.replace(hour=start.hour, minute=start.minute,
                         second=0, microsecond=0)
        if dt > now:
            return dt
    d = now.date() + _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return _dt.datetime.combine(d, TRADING_AM[0])


__all__ = ["is_trading_session", "next_session", "TRADING_AM", "TRADING_PM"]
