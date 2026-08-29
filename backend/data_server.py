"""真实行情后端服务（轻量 HTTP API，零新增依赖）。

把项目已有的 ``DataGateway``（AKShare/BaoStock 真实源 + 本地 SQLite 缓存 + 演示降级）
包成一个可被 Streamlit 终端、脚本或其它前端直连的 HTTP 服务：

    GET /api/health                后端与数据源健康度（akshare 是否可用、最近降级原因）
    GET /api/history?symbol=&start=&end=&asset=&force_refresh=
                                    历史 OHLCV（真实 AKShare 数据，自动走缓存）
    GET /api/quote?symbol=&asset=   实时快照（AKShare 实时盘口，降级到最后收盘）
    GET /api/quote_batch?symbols=   批量实时快照（逗号分隔）
    GET /api/cache                  已落库的真实数据清单（便于离线回测前核对覆盖）
    GET /                          API 说明页

设计取舍：
- 用标准库 ``http.server``（ThreadingHTTPServer）而非 FastAPI/Flask，坚守核心零重依赖纪律；
  真实数据能力完全复用 ``DataGateway``，不重复造轮子。
- 启动可选 ``--warm`` 预热 watchlist 的真实数据进本地缓存，让终端打开即有真实行情可用；
  可选 ``--live-poll N`` 每 N 秒刷新一次 watchlist 的实时报价并缓存到内存，供 /api/quote 秒回。
- 断网 / 接口变更时 ``DataGateway`` 自动降级演示数据，并在响应里如实标注 ``source=demo``，
  绝不把假数据冒充真实行情返回。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------- 让脚本在任意 cwd 下都能 import 到 lianghua 包 ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lianghua.data.gateway import DataGateway  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [data-backend] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_backend")

DEFAULT_PORT = 8600  # 避开 8501(终端)/8600 与其它量化软件冲突


class Backend:
    """真实行情后端：持有网关、预热缓存、可选实时轮询。"""

    def __init__(self, watchlist_path: str | None = None,
                 real_timeout: float = 10.0):
        self.gw = DataGateway()
        self.real_timeout = real_timeout
        self.watchlist = self._load_watchlist(watchlist_path)
        # 内存实时报价缓存：symbol -> {"price":float,"source":str,"ts":float}
        self.live_cache: dict[str, dict] = {}
        self._preclose_cache: dict[str, float] = {}  # symbol -> 昨收（1h 缓存，避免频繁回源）
        self.live_lock = threading.Lock()
        self._stop = threading.Event()

    @staticmethod
    def _load_watchlist(path: str | None) -> list[dict]:
        p = Path(path) if path else ROOT / "backend" / "watchlist.json"
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("items", []) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("watchlist 读取失败: %s", exc)
            return []

    # ---------- 历史行情（真实 AKShare 数据） ----------
    def get_history(self, symbol: str, start: str, end: str,
                    asset: str | None = None, force_refresh: bool = False) -> dict:
        df = self.gw.fetch(symbol, start, end, asset=asset,
                           timeout=self.real_timeout, force_refresh=force_refresh)
        recs = json.loads(df.drop(columns=["source"], errors="ignore").to_json(
            orient="records", date_format="iso"))
        return {
            "symbol": symbol,
            "asset": asset or "auto",
            "source": self.gw.last_source,          # akshare / cache / baostock / demo
            "was_demo": self.gw.last_was_demo,
            "rows": len(recs),
            "start": start, "end": end,
            "data": recs,
        }

    # ---------- 实时快照 ----------
    def get_quote(self, symbol: str, asset: str | None = None) -> dict:
        # 若开启实时轮询且有新鲜缓存则直接返回，否则按需取
        if not self._stop.is_set():
            with self.live_lock:
                cached = self.live_cache.get(symbol)
            if cached and (time.time() - cached.get("ts", 0)) < 30:
                return {**cached, "cached": True}
        # 取实时价（优先 AKShare 实时盘口，失败降级到最后收盘/演示价）
        try:
            q = self.gw.live_quote(symbol, asset=asset)
            price = q["price"]
            source = q["source"]
            was_demo = self.gw.last_was_demo
        except Exception:  # noqa: BLE001
            price, source, was_demo = float("nan"), "none", False
        # 取昨收（用于算日内涨跌幅），带 1h 缓存避免每条报价都回源历史
        preclose = self._preclose_cache.get(symbol)
        if preclose is None:
            try:
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
                df = self.gw.fetch(symbol, start, end, asset=asset)
                if df is not None and len(df) >= 2:
                    preclose = float(df["close"].iloc[-2])
                elif df is not None and len(df) == 1:
                    preclose = float(df["close"].iloc[-1])
                else:
                    preclose = None
            except Exception:  # noqa: BLE001
                preclose = None
            if preclose is not None:
                self._preclose_cache[symbol] = preclose
        change = (price - preclose) if (preclose and price == price) else None
        pct = (change / preclose) if (change is not None and preclose) else None
        out = {
            "symbol": symbol, "price": price, "source": source,
            "was_demo": was_demo, "preclose": preclose,
            "change": change, "pct_change": pct,
            "ts": time.time(), "cached": False,
        }
        with self.live_lock:
            self.live_cache[symbol] = out
        return out

    def get_quote_batch(self, symbols: list[str]) -> list[dict]:
        out = []
        for s in symbols:
            s = s.strip()
            if not s:
                continue
            out.append(self.get_quote(s))
        return out

    # ---------- 启动预热 / 实时轮询 ----------
    def warm(self):
        """启动预热：把 watchlist 的真实数据拉进本地缓存（断网则降级演示并标注）。"""
        if not self.watchlist:
            logger.info("watchlist 为空，跳过预热")
            return
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        ok = demo = 0
        for it in self.watchlist:
            sym = it["symbol"]
            asset = it.get("asset")
            try:
                df = self.gw.fetch(sym, start, end, asset=asset,
                                   timeout=self.real_timeout)
                if self.gw.last_was_demo:
                    demo += 1
                    logger.warning("预热 %s -> 降级为演示数据 (source=%s)",
                                   sym, self.gw.last_source)
                else:
                    ok += 1
                    logger.info("预热 %s -> 真实数据 OK (%d 行, source=%s)",
                                sym, len(df), self.gw.last_source)
            except Exception as exc:  # noqa: BLE001
                demo += 1
                logger.warning("预热 %s 失败: %s", sym, exc)
        logger.info("预热完成：真实 %d / 降级 %d", ok, demo)

    def live_poll_loop(self, interval: float):
        """后台线程：每隔 interval 秒刷新一次 watchlist 实时报价到内存缓存。"""
        logger.info("实时轮询已启动，间隔 %.0f 秒", interval)
        while not self._stop.is_set():
            for it in self.watchlist:
                if self._stop.is_set():
                    break
                sym = it["symbol"]
                try:
                    # 复用 get_quote：统一产出 preclose/change/pct_change 等富字段，
                    # 避免手动拼旧格式字典导致 /api/quote 命中缓存时丢失涨跌信息。
                    self.get_quote(sym, asset=it.get("asset"))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("实时轮询 %s 失败: %s", sym, exc)
            self._stop.wait(interval)

    # ---------- 健康度 ----------
    def health(self) -> dict:
        """轻量级健康探针：仅读取内存态字段，绝不触 SQLite / 网络。

        终端会以 2~5s 超时频繁轮询本端点判断后端存活；若此处做任何 I/O
        （如 list_cached 的 GROUP BY + 逐标的 COUNT），在 SQLite 被实时轮询线程
        占用时会偶发 >timeout 的延迟，导致探针误判后端离线、整页降级。
        已落库标的清单请走专有的 /api/cache 端点（本就不该出现在心跳里）。
        """
        try:
            import importlib.util as u
            akshare_ok = u.find_spec("akshare") is not None
        except Exception:  # noqa: BLE001
            akshare_ok = False
        return {
            "status": "up",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "akshare_available": akshare_ok,
            "last_source": self.gw.last_source,
            "last_was_demo": self.gw.last_was_demo,
            "recent_errors": self.gw.last_warnings()[-5:],
            "live_cached": len(self.live_cache),
            "watchlist_size": len(self.watchlist),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "LianghuaDataBackend/1.0"

    # 让日志更安静
    def log_message(self, fmt, *args):  # noqa: A003
        logger.info("HTTP " + fmt, *args)

    def _send(self, code: int, payload):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _query(self) -> dict:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        q = dict(urllib.parse.parse_qsl(parsed.query))
        backend: Backend = self.server.backend  # type: ignore[attr-defined]
        try:
            if route in ("/", "/api"):
                self._send(200, {
                    "service": "Lianghua 真实行情后端",
                    "endpoints": [
                        "/api/health",
                        "/api/history?symbol=600519.SH&start=2024-01-01&end=2024-03-31&asset=stock",
                        "/api/quote?symbol=600519.SH&asset=stock",
                        "/api/quote_batch?symbols=600519.SH,510300.SH",
                        "/api/stream?symbols=600519.SH,510300.SH&interval=2",
                        "/api/cache",
                        "/api/refresh?symbol=600519.SH&start=2024-01-01&end=2024-03-31&asset=stock",
                    ],
                })
                return
            if route == "/api/health":
                self._send(200, backend.health())
                return
            if route == "/api/history":
                sym = q.get("symbol")
                if not sym:
                    self._err(400, "缺少 symbol 参数")
                    return
                start = q.get("start") or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                end = q.get("end") or datetime.now().strftime("%Y-%m-%d")
                force = str(q.get("force_refresh", "")).lower() in ("1", "true", "yes")
                try:
                    self._send(200, backend.get_history(
                        sym, start, end, asset=q.get("asset"), force_refresh=force))
                except Exception as exc:  # noqa: BLE001
                    self._err(500, f"历史行情获取失败: {exc}")
                return
            if route == "/api/quote":
                sym = q.get("symbol")
                if not sym:
                    self._err(400, "缺少 symbol 参数")
                    return
                try:
                    self._send(200, backend.get_quote(sym, asset=q.get("asset")))
                except Exception as exc:  # noqa: BLE001
                    self._err(500, f"实时行情获取失败: {exc}")
                return
            if route == "/api/quote_batch":
                syms = [s for s in q.get("symbols", "").split(",") if s.strip()]
                if not syms:
                    self._err(400, "缺少 symbols 参数（逗号分隔）")
                    return
                self._send(200, backend.get_quote_batch(syms))
                return
            if route == "/api/stream":
                # SSE 实时推送流：后端主动按 interval 秒推送批量报价，
                # 替代终端侧 15s 轮询。断流/客户端断开即结束，绝不把假数据冒充真实行情。
                syms = [s.strip() for s in q.get("symbols", "").split(",") if s.strip()]
                sym = q.get("symbol", "").strip()
                if not syms and sym:
                    syms = [sym]
                if not syms:
                    self._err(400, "缺少 symbols 或 symbol 参数")
                    return
                interval = float(q.get("interval", 2) or 2)
                self._stream_quotes(backend, syms, interval)
                return
            if route == "/api/cache":
                self._send(200, backend.gw.list_cached())
                return
            if route == "/api/refresh":
                # 强制回源刷新某标的真实历史行情进本地缓存（绕过缓存命中）
                sym = q.get("symbol")
                if not sym:
                    self._err(400, "缺少 symbol 参数")
                    return
                start = q.get("start") or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                end = q.get("end") or datetime.now().strftime("%Y-%m-%d")
                try:
                    self._send(200, backend.get_history(
                        sym, start, end, asset=q.get("asset"), force_refresh=True))
                except Exception as exc:  # noqa: BLE001
                    self._err(500, f"强制刷新失败: {exc}")
                return
            self._err(404, f"未知路由: {route}")
        except Exception as exc:  # noqa: BLE001
            self._err(500, f"服务器内部错误: {exc}")

    def do_OPTIONS(self):  # noqa: N802  - 支持浏览器跨域预检
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _stream_quotes(self, backend: "Backend", symbols: list[str], interval: float) -> None:
        """SSE 事件流：按 interval 秒持续推送 watchlist 批量报价，直到客户端断开。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")  # 禁用反向代理缓冲，保证秒级推送
        self.end_headers()
        try:
            while True:
                try:
                    payload = json.dumps({
                        "event": "quote",
                        "ts": time.time(),
                        "data": backend.get_quote_batch(symbols),
                    }, ensure_ascii=False, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:  # noqa: BLE001 - 单帧异常不致命，继续推下一帧
                    break
                time.sleep(interval)
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser(description="Lianghua 真实行情后端服务")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口 (默认 {DEFAULT_PORT})")
    ap.add_argument("--no-warm", action="store_true", help="启动不预热 watchlist")
    ap.add_argument("--live-poll", type=float, default=0,
                   help="实时报价轮询间隔秒数 (0=关闭，建议 30)")
    ap.add_argument("--timeout", type=float, default=10.0,
                   help="真实源单次取数超时(秒)，超时自动降级演示")
    ap.add_argument("--watchlist", default=None, help="watchlist.json 路径")
    args = ap.parse_args()

    backend = Backend(watchlist_path=args.watchlist, real_timeout=args.timeout)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.backend = backend  # type: ignore[attr-defined]

    if not args.no_warm:
        logger.info("启动预热真实行情缓存 ...")
        threading.Thread(target=backend.warm, daemon=True).start()

    poll_thread = None
    if args.live_poll and args.live_poll > 0:
        poll_thread = threading.Thread(
            target=backend.live_poll_loop, args=(args.live_poll,), daemon=True)
        poll_thread.start()

    logger.info("真实行情后端已启动: http://%s:%d  (--live-poll=%s)",
                args.host, args.port, args.live_poll)
    logger.info("快速自检: curl http://localhost:%d/api/health", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭 ...")
    finally:
        backend._stop.set()
        server.shutdown()
        server.server_close()
        logger.info("后端已停止")


if __name__ == "__main__":
    main()
