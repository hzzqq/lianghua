"""数据网关取数超时测试：超时必须真的生效，且原因如实可观测。

覆盖的真实故障（本轮修复前会 100% 复现）：

1. ``_with_timeout`` 用 ``with ThreadPoolExecutor(...)`` 包住 ``fut.result(timeout=)``，
   而 ``__exit__`` 会 ``shutdown(wait=True)`` 去 join 那个仍在跑的工作线程 ——
   于是 TimeoutError 抛出后照样卡在退出处，**声称的超时保护完全失效**。
   实测：无网环境下 baostock 的 ``rs.next()`` 阻塞十几分钟，``fetch(timeout=15)``
   一直不返回，``LiveEngine.step()`` 调仓循环整体僵死且监控看不到任何 error；
   项目全量测试也因此从几分钟拖到 1 小时以上跑不完。

2. 失败原因记录写成了 ``type(ex).__name__``，而 ``ex`` 是 ``with ... as ex`` 绑定的
   **线程池对象**而非异常 —— 排障时只能看到 "ThreadPoolExecutor: xxx"，
   把上一轮刚做好的"失败原因可观测"又变回了噪声。
"""
import threading
import time

import numpy as np
import pandas as pd
import pytest

from lianghua.data.gateway import DataGateway


@pytest.fixture
def gw():
    return DataGateway(cache_db=":memory:")


def _ohlc():
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    close = 100.0 * np.cumprod(1.0 + np.random.default_rng(1).normal(0.001, 0.01, len(dates)))
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6})


def test_hanging_source_times_out_instead_of_blocking(gw):
    """真实源卡死不返回时，fetch 必须在 timeout 附近返回并降级，而不是无限期挂起。"""
    release = threading.Event()
    entered = threading.Event()

    def hang(symbol, start, end, asset=None):
        entered.set()
        release.wait(120)  # 模拟无网时 baostock rs.next() 的长阻塞
        return _ohlc()

    gw._from_akshare = hang
    gw._from_baostock = hang
    t0 = time.monotonic()
    try:
        df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                      timeout=0.5, retries=0, backoff=0)
        elapsed = time.monotonic() - t0
    finally:
        release.set()  # 放掉后台守护线程，避免拖累后续用例

    assert entered.is_set(), "真实源应当被调用过"
    assert elapsed < 20, (
        f"fetch 耗时 {elapsed:.1f}s 远超 timeout=0.5s —— 超时保护未生效（线程池 join 反噬）"
    )
    assert gw.last_was_demo, "真实源超时后应降级为演示数据"
    assert df["source"].iloc[0] == "demo"


def test_timeout_reason_is_observable(gw):
    """超时降级必须在 last_warnings 里如实写明是超时，而不是含糊的"无有效数据"。"""
    release = threading.Event()

    def hang(symbol, start, end, asset=None):
        release.wait(120)
        return _ohlc()

    gw._from_akshare = hang
    gw._from_baostock = hang
    try:
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                 timeout=0.3, retries=0, backoff=0)
    finally:
        release.set()

    warns = gw.last_warnings()
    assert any("TimeoutError" in w for w in warns), \
        f"超时原因应含 TimeoutError，实际: {warns}"
    assert not any("ThreadPoolExecutor" in w for w in warns), \
        f"记录的应是异常类型而不是线程池对象，实际: {warns}"


def test_source_exception_type_is_recorded(gw):
    """真实源抛异常时，记录的类型名必须是异常本身（此前误记成 ThreadPoolExecutor）。"""

    def boom(symbol, start, end, asset=None):
        raise ConnectionError("dns lookup failed")

    gw._from_akshare = boom
    gw._from_baostock = boom
    gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
             timeout=5, retries=0, backoff=0)
    warns = gw.last_warnings()
    assert any("ConnectionError" in w and "dns lookup failed" in w for w in warns), \
        f"应记录真实异常类型与消息，实际: {warns}"


def test_timeout_does_not_break_fast_source(gw):
    """超时改造不得影响正常路径：快源应照常返回真实数据。"""
    gw._from_akshare = lambda symbol, start, end, asset=None: _ohlc()
    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                  timeout=5, retries=0, backoff=0)
    assert not gw.last_was_demo
    assert gw.last_source == "akshare"
    assert len(df) > 0


def test_timeout_is_not_retried_into_a_request_storm(gw):
    """卡死的源不得被重试放大：一次 fetch 里每个源最多只发一次真实请求。

    放弃线程后那条请求仍占着连接在跑，立刻重试等于对同一个已证明卡死的端点
    再压一条连接；``retries`` 是为"快速失败的瞬时抖动"准备的，不是为超时准备的。
    """
    release = threading.Event()
    calls: list[str] = []

    def ak_source(symbol, start, end, asset=None):
        calls.append("ak")
        release.wait(120)
        return _ohlc()

    def bs_source(symbol, start, end, asset=None):
        calls.append("bs")
        release.wait(120)
        return _ohlc()

    gw._from_akshare = ak_source
    gw._from_baostock = bs_source
    try:
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                 timeout=0.3, retries=3, backoff=0)
    finally:
        release.set()

    # akshare 1 次 + baostock 1 次；retries=3 不得把它放大成 (3+1)*2=8 次。
    # 只断言请求次数：它是确定性的，不像墙钟耗时那样在满载的全量测试里飘。
    assert calls == ["ak", "bs"], f"卡死源被重试放大为 {calls}"
    assert gw.last_was_demo


def test_hung_source_is_skipped_until_it_actually_returns(gw):
    """跨 fetch 的在途守卫：源还卡着就直接降级，它一返回就立刻恢复可用。

    没有这道守卫时，LiveEngine 每轮轮询都会对同一个卡死的源新起一条守护线程，
    线程与 socket 无上限堆积；实测 test_paper_end_to_end（2 标的 x 2 步）
    因此空等 230s。
    """
    release = threading.Event()
    calls: list[str] = []

    def ak_source(symbol, start, end, asset=None):
        calls.append("ak")
        release.wait(120)
        return _ohlc()

    def bs_source(symbol, start, end, asset=None):
        calls.append("bs")
        release.wait(120)
        return _ohlc()

    gw._from_akshare = ak_source
    gw._from_baostock = bs_source
    try:
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                 timeout=0.3, retries=0, backoff=0)
        assert calls == ["ak", "bs"]

        # 第二次 fetch：两个源都还卡着 -> 一次新请求都不该发，且要立即降级。
        # 故意把 timeout 放大到 5s：守卫生效则几乎零耗时，失效则要等满两个 5s，
        # 于是 2s 的判据有足够余量，不会在满载的全量测试里误报。
        t0 = time.monotonic()
        gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                 timeout=5, retries=0, backoff=0, force_refresh=True)
        assert calls == ["ak", "bs"], "卡死期间不应再对该源发起新请求"
        assert time.monotonic() - t0 < 2, "在途守卫应当立即降级而不是再等一个 timeout"
        assert gw.last_was_demo
        assert any("仍未返回" in w for w in gw.last_warnings())
    finally:
        release.set()

    # 卡死的线程真的返回后，守卫自愈：同一个源重新可用（不依赖任何冷却常量）
    for _ in range(300):
        if not gw._abandoned_alive("ak_source"):
            break
        time.sleep(0.02)
    df = gw.fetch("600000.SH", "2024-01-01", "2024-01-10", asset="stock",
                  timeout=5, retries=0, backoff=0, force_refresh=True)
    assert calls.count("ak") == 2, "线程返回后该源应被重新调用"
    assert not gw.last_was_demo, "卡死线程返回后该源应恢复可用（守卫需自愈）"
    assert gw.last_source == "akshare"
    assert len(df) > 0
