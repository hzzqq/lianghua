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


def main() -> int:
    ap = argparse.ArgumentParser(description="实盘定时调仓")
    ap.add_argument("--config", default=os.path.join(ROOT, "live_cron.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验配置与交易时段，不建真实引擎")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(json.dumps({"error": "config not found: %s" % args.config},
                         ensure_ascii=False))
        return 1

    config = load_config(args.config)
    session = is_trading_session()

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
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
