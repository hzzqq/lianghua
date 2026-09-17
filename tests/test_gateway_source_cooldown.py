"""源级失败冷却（熔断）测试。

R19 Backlog：断网/被墙时 AKShare/腾讯会「立即抛错」（ProxyError/DNS/501，毫秒级），
每次 fetch 都把整条源链快速失败+重试跑一遍（单符号 ~11s，冷启动首屏空窗）。
冷却熔断：同一源连续快速失败达阈值后短时间内直接跳过（指数退避，封顶）。

纪律边界（rules.md §2，本轮已获授权）：
- fetch 源优先级顺序完全不动（缓存→CSV→akshare→腾讯→baostock→demo）；
- 超时类失败不进冷却（由 _with_timeout 在途守卫自愈管理）；
- 任一尝试成功即清零；CSV/缓存路径不经过熔断；force_refresh 绕过冷却。
"""
import threading
import time

import numpy as np
import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw():
    # :memory: 即可：本文件不依赖缓存持久性（全部 fetch 用 force_refresh 绕过缓存）
    return DataGateway(cache_db=":memory:")


def _real_df():
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    close = 100.0 * np.cumprod(1.0 + np.full(len(dates), 0.001))
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6})


def _fail_source(gw, attr):
    def always_fail(*_a, **_k):
        raise ConnectionError("simulated offline: proxy 501")
    always_fail.__name__ = attr
    setattr(gw, attr, always_fail)


def test_cooldown_skips_source_after_repeated_fast_failures(gw):
    """连续快速失败达阈值后，下一次 fetch 不再调用该源，且跳过必须留痕。"""
    calls = {"n": 0}

    def fail_ak(*_a, **_k):
        calls["n"] += 1
        raise ConnectionError("offline")

    fail_ak.__name__ = "fail_ak"
    gw._from_akshare = fail_ak
    gw._from_tencent = lambda *a, **k: _real_df()  # 腾讯可用：整体取数仍走真实源

    # 第 1 次：akshare 快速失败 1 次（未达阈值 2），腾讯兜底成功
    gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
             retries=0, backoff=0, force_refresh=True)
    assert calls["n"] == 1
    # 第 2 次：akshare 快速失败第 2 次 → 进入冷却
    gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
             retries=0, backoff=0, force_refresh=True)
    assert calls["n"] == 2
    assert gw._source_cooling("fail_ak")
    # 第 3 次：冷却中的 akshare 被跳过，不再调用；腾讯照常返回真实数据
    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                  retries=0, backoff=0, force_refresh=True)
    assert calls["n"] == 2, "冷却中的源不应再被调用"
    assert gw.last_source == "tencent", "冷却只跳过死源，不得影响其它真实源"
    assert not gw.last_was_demo
    assert any("冷却中" in w and "fail_ak" in w for w in gw.last_warnings()), \
        "跳过行为必须留痕可观测"


def test_success_resets_streak_and_cooldown(gw):
    """任一尝试成功即清零失败连击并解除冷却（源恢复即恢复使用）。"""
    state = {"fails_left": 1}

    def flaky(*_a, **_k):
        if state["fails_left"] > 0:
            state["fails_left"] -= 1
            raise ConnectionError("transient")
        return _real_df()

    flaky.__name__ = "flaky"
    gw._from_akshare = flaky
    gw._from_tencent = lambda *a, **k: _real_df()

    # 第 1 次 fetch：flaky 快速失败 1 次（未达阈值）
    gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
             retries=0, backoff=0, force_refresh=True)
    assert gw._fail_streak["flaky"] == 1
    # 第 2 次 fetch：flaky 成功 → 连击清零、无冷却
    gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
             retries=0, backoff=0, force_refresh=True)
    assert gw._fail_streak["flaky"] == 0
    assert not gw._source_cooling("flaky")

    # 状态机直测：满连击进入冷却后，一次成功即解除
    gw._record_source_fast_fail("flaky")
    gw._record_source_fast_fail("flaky")
    assert gw._source_cooling("flaky")
    gw._record_source_success("flaky")
    assert not gw._source_cooling("flaky")
    assert gw._fail_streak["flaky"] == 0


def test_cooldown_applies_even_with_force_refresh(gw):
    """冷却对 force_refresh 同样生效（与在途守卫一致的保护性闸门）。

    否则终端 /api/refresh 与 UI 手动刷新会在冷却期内持续轰击刚被证明死亡的源；
    冷却基数仅 30s 且源线程一旦正常完成即清零，手动刷新最多等一个周期。
    """
    calls = {"n": 0}

    def fail_ak(*_a, **_k):
        calls["n"] += 1
        raise ConnectionError("offline")

    fail_ak.__name__ = "fail_ak"
    gw._from_akshare = fail_ak
    gw._from_tencent = lambda *a, **k: _real_df()

    gw._record_source_fast_fail("fail_ak")
    gw._record_source_fast_fail("fail_ak")
    assert gw._source_cooling("fail_ak")

    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                  retries=0, backoff=0, force_refresh=True)
    assert calls["n"] == 0, "冷却期内 force_refresh 也不应轰击死源"
    assert gw.last_source == "tencent", "冷却只跳过死源，整体取数仍走其它真实源"


def test_timeout_failures_do_not_arm_cooldown(gw):
    """超时类失败不进冷却（在途守卫负责自愈）；线程返回后源立即可用。"""
    release = threading.Event()

    def hang(*_a, **_k):
        release.wait(60)
        return _real_df()

    hang.__name__ = "hang"
    gw._from_akshare = hang
    gw._from_tencent = lambda *a, **k: _real_df()

    try:
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                 timeout=0.2, retries=0, backoff=0, force_refresh=True)
    finally:
        release.set()
    assert not gw._source_cooling("hang"), "超时失败不应触发冷却（在途守卫负责）"

    # 被放弃的线程返回后：在途守卫自愈，akshare 立即恢复可用
    deadline = time.time() + 5
    while gw._abandoned_alive("hang"):
        assert time.time() < deadline, "在途守卫未自愈"
        time.sleep(0.02)
    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                  timeout=5, retries=0, backoff=0, force_refresh=True)
    assert gw.last_source == "akshare", "源恢复后应立即重新可用"
    assert len(df) > 0


def test_cooldown_is_per_source(gw):
    """冷却按源独立：A 源熔断不影响 B 源照常取数。"""
    def fail_ak(*_a, **_k):
        raise ConnectionError("offline")

    fail_ak.__name__ = "fail_ak"

    def tx_ok(*_a, **_k):
        return _real_df()

    tx_ok.__name__ = "tx_ok"
    gw._from_akshare = fail_ak
    gw._from_tencent = tx_ok

    for _ in range(2):
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                 retries=0, backoff=0, force_refresh=True)
    assert gw._source_cooling("fail_ak")
    assert not gw._source_cooling("tx_ok"), "其它源不应被连带冷却"


def test_exponential_backoff_capped(gw):
    """冷却时长指数退避且封顶：n=2→30s, n=3→60s, n=10→封顶 300s。"""
    base = DataGateway.COOLDOWN_BASE_SECONDS
    cap = DataGateway.COOLDOWN_MAX_SECONDS
    gw2 = DataGateway(cache_db=":memory:")
    for n in range(1, 11):
        gw2._record_source_fast_fail("src")
        if n < DataGateway.COOLDOWN_AFTER_FAST_FAILS:
            assert not gw2._source_cooling("src")
            continue
        with gw2._cooldown_lock:
            until = gw2._cooldown_until["src"]
        remaining = until - time.monotonic()
        expect = min(base * (2 ** (n - DataGateway.COOLDOWN_AFTER_FAST_FAILS)), cap)
        assert expect - 1 <= remaining <= expect, f"n={n} 冷却时长不符"
