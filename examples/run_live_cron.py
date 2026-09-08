# -*- coding: utf-8 -*-
"""实盘定时调仓脚本（第十三轮·深化方向二：定时调仓 automation）。

由 automation / 任务计划程序周期性调用。读取 live_cron.json 配置，对每个账户
执行一次 step；自动跳过非交易时段；msvcrt 文件锁防重入；状态写入
live_cron_state.json。

用法：
    python examples/run_live_cron.py [--config live_cron.json]
                                    [--dry-run] [--once]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from lianghua.execution.schedule import is_trading_session
from lianghua.execution.router import make_multi_engine
from lianghua.execution.live import make_live_engine
from lianghua.execution.notify import build_notifier


def _lock_path(config_path: str) -> str:
    base = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(tempfile.gettempdir(), "lianghua_cron_%s.lock" % base)


def acquire_lock(lock_path: str):
    """Windows 文件锁（msvcrt.locking 非阻塞）。返回 fd 或 None（已在运行）。"""
    try:
        import msvcrt
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return fd
    except Exception:
        return None


def release_lock(fd) -> None:
    if fd is None:
        return
    try:
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)
    except Exception:
        pass


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_summary(res: dict, limit: int = 8) -> str:
    """把单次调仓结果压成中文文本，供微信推送。"""
    lines = ["【量化框架·定时调仓】"]
    if res.get("skipped"):
        lines.append("跳过：%s" % res.get("reason", ""))
        return "\n".join(lines)
    if res.get("mode") == "multi":
        accs = (res.get("status") or {}).get("accounts", {})
        for name, st in accs.items():
            acts = st.get("actions", []) if isinstance(st, dict) else []
            n_trade = sum(1 for a in acts if a.get("action") == "trade")
            n_exit = sum(1 for a in acts if a.get("action") == "exit")
            lines.append("- 账户%s：成交%d 平仓%d" % (name, n_trade, n_exit))
        rb = res.get("rebalance")
        if rb:
            lines.append("再平衡：总额%.0f，%d个账户" %
                         (rb.get("total", 0), len(rb.get("plan", []))))
    else:
        acts = (res.get("status") or {}).get("actions", [])
        n_trade = sum(1 for a in acts if a.get("action") == "trade")
        n_exit = sum(1 for a in acts if a.get("action") == "exit")
        lines.append("单账户：成交%d 平仓%d" % (n_trade, n_exit))
    return "\n".join(lines)


def run_once(config: dict) -> dict:
    ok, why = is_trading_session()
    if not ok:
        return {"skipped": True, "reason": why, "ts": str(_dt.datetime.now())}
    notifier = build_notifier(config.get("notify"))
    if "accounts" in config:
        eng = make_multi_engine(config)
        if notifier:
            eng.notify = notifier.push
        status = eng.step()
        res = {"skipped": False, "mode": "multi", "status": status}
        if config.get("rebalance"):
            res["rebalance"] = eng.rebalance(config.get("target_weights"))
        if notifier:
            notifier.push(_format_summary(res))
        return res
    eng = make_live_engine(
        config.get("broker", "paper"),
        config.get("strategy", "sma_cross"),
        config.get("symbols", ["600519.SH"]),
        capital=config.get("capital", 1_000_000.0),
        lookback=config.get("lookback", 120),
        live=config.get("live", False),
        freq=config.get("freq", "daily"),
        risk_limits=config.get("risk"),
        sl_pct=config.get("sl_pct"),
        tp_pct=config.get("tp_pct"),
        trailing_pct=config.get("trailing_pct"),
    )
    if notifier:
        eng.notify = notifier.push
    status = eng.step()
    res = {"skipped": False, "mode": "single", "status": status}
    if notifier:
        notifier.push(_format_summary(res))
    return res


def _write_state(result: dict, state_path: str | None = None) -> None:
    """把单次执行结果追加进 live_cron_state.json（保留最近 50 条）。"""
    if state_path is None:
        state_path = os.path.join(ROOT, "live_cron_state.json")
    hist = []
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []
    hist.append(result)
    hist = hist[-50:]
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"warn": "写入 cron 状态失败: %s" % exc},
                         ensure_ascii=False))


def check_data_sanity(config: dict, lookback_days: int = 30,
                      refuse_ratio: float = 1/3) -> dict:
    """实盘数据健全性门禁：统计配置标的的演示(假)数据占比，超阈值拒绝启动。

    接入点即 ``DataGateway``：对配置里的每个标的做一次普通 fetch（命中真实缓存即视为真实，
    演示缓存会被网关自动绕过并重试真实源），读取 ``last_was_demo`` 判定该标的当前是否可信任。
    这样能拦住「指数代码无日线接口 / 期货主连接口偶发失败 / 网络抖动」等导致的静默假数据，
    避免用随机游走跑出假回测、假调仓结论（即简历所述「demo 占比超阈值拒绝启动」门禁）。

    返回 ``{"ok", "demo", "ratio", "total", "refuse_ratio"}``：
    - ``ratio > refuse_ratio`` 时 ``ok=False``（应拒绝启动）；
    - 任意标的是演示数据都会在 ``demo`` 中列出，供告警（即便未达拒绝阈值）。
    """
    from lianghua.data.gateway import DataGateway
    from lianghua.core.assets import detect_asset_type
    gw = DataGateway()
    symbols = list(config.get("symbols", []))
    for acc in config.get("accounts", []):
        symbols.extend(acc.get("symbols", []))
    symbols = list(dict.fromkeys(symbols))  # 去重保序
    if not symbols:
        return {"ok": True, "demo": [], "ratio": 0.0, "total": 0,
                "refuse_ratio": refuse_ratio}
    today = _dt.date.today()
    start = (today - _dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    demo = []
    for s in symbols:
        at = detect_asset_type(s)
        try:
            # 门禁只跑一次/每轮，放宽到 60s：慢源(如 Sina 期货主连 2000+ 行)
            # 不至于因 15s 默认超时被判成演示(假)数据而误拒。
            gw.fetch(s, start, end, asset=at, timeout=60.0)
        except Exception:
            pass
        if gw.last_was_demo:
            demo.append(s)
    ratio = (len(demo) / len(symbols)) if symbols else 0.0
    return {"ok": ratio <= refuse_ratio, "demo": demo, "ratio": ratio,
            "total": len(symbols), "refuse_ratio": refuse_ratio}


def main() -> int:
    ap = argparse.ArgumentParser(description="实盘定时调仓")
    ap.add_argument("--config", default=os.path.join(ROOT, "live_cron.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验配置与交易时段，不建真实引擎")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-gate", action="store_true",
                    help="跳过数据健全性门禁(仅紧急排障用)")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(json.dumps({"error": "config not found: %s" % args.config},
                         ensure_ascii=False))
        return 1

    config = load_config(args.config)
    session = is_trading_session()
    notifier = build_notifier(config.get("notify")) if not args.dry_run else None

    # ---- 数据健全性门禁（P0：杜绝用假数据跑出假结论）----
    if not args.dry_run and not args.no_gate:
        ds = config.get("data_sanity", {})
        sane = check_data_sanity(
            config,
            lookback_days=int(ds.get("lookback_days", 30)),
            refuse_ratio=float(ds.get("demo_refuse_ratio", 1/3)),
        )
        if not sane["ok"]:
            reason = ("数据门禁拦截：%d/%d 标的为演示(假)数据(占比%.0f%%>%.0f%%)，拒绝启动"
                      % (len(sane["demo"]), sane["total"],
                         sane["ratio"] * 100, sane["refuse_ratio"] * 100))
            print(json.dumps({"skipped": True, "reason": reason,
                              "demo_symbols": sane["demo"]}, ensure_ascii=False))
            if notifier:
                notifier.push("【量化框架·定时调仓】被数据门禁拦截：%s。假数据标的：%s"
                              % (reason, ", ".join(sane["demo"])))
            _write_state({"skipped": True, "reason": reason,
                          "demo_symbols": sane["demo"], "ts": str(_dt.datetime.now())})
            return 2
        if sane["demo"]:
            print(json.dumps({"data_sanity": "warn", "demo_symbols": sane["demo"],
                              "ratio": sane["ratio"]}, ensure_ascii=False))
            if notifier:
                notifier.push("【量化框架·定时调仓】数据告警：%d 个标的为演示(假)数据(%s)，占比%.0f%%，未超阈值仍启动。"
                              % (len(sane["demo"]), ", ".join(sane["demo"]),
                                 sane["ratio"] * 100))

    if args.dry_run:
        summary = {
            "dry_run": True,
            "session": {"trading": session[0], "desc": session[1]},
            "accounts": [a.get("name") for a in config.get("accounts", [])]
            or [config.get("broker", "paper")],
            "symbols": config.get("symbols", []),
        }
        print(json.dumps(summary, ensure_ascii=False, default=str))
        return 0

    lock = acquire_lock(_lock_path(args.config))
    if lock is None:
        print(json.dumps({"skipped": True, "reason": "已有实例在运行(锁)"},
                         ensure_ascii=False))
        return 0
    try:
        result = run_once(config)
    finally:
        release_lock(lock)

    _write_state(result)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
