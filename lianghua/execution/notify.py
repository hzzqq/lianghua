"""实盘信号推送（迭代30）。

把策略/回测产生的信号事件推送到多个渠道：
- log：本地日志文件（始终可用，零依赖）
- webhook：HTTP POST 到指定 URL（依赖 requests，缺失则优雅降级为日志告警）

设计对齐 execution/brokers 的 BaseBroker 风格：
- push(event) 返回结构化结果 dict，并记录到日志。
- 支持 attach/detach 多 Channel，便于扩展（邮件/企微/钉钉）。
"""
from __future__ import annotations

import json
import logging
import os
import datetime as _dt
from typing import Callable

__all__ = ["SignalNotifier", "LogChannel", "WebhookChannel", "Channel",
           "WeChatNotifier", "WeComAppNotifier", "NotifyHub", "build_notifier"]


class Channel:
    """推送渠道基类。"""

    name: str = "base"

    def send(self, event: dict) -> dict:
        raise NotImplementedError

    def healthy(self) -> bool:
        return True


class LogChannel(Channel):
    """本地日志渠道：写入文件（或内存缓冲）。始终可用。

    使用独立的 FileHandler 而非 logging.basicConfig，避免被其它模块
    已配置的 root logger 干扰（basicConfig(force=False) 在 root 已有
    handler 时不会生效，导致日志实际未落盘）。
    """

    name = "log"

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        self.buffer: list[str] = []
        self._handler: logging.Handler | None = None
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
            self._logger = logging.getLogger("signal_notify.%s" % id(self))
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False  # 不向 root 冒泡，避免重复/错位输出
            self._handler = logging.FileHandler(log_file, encoding="utf-8")
            self._handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._logger.addHandler(self._handler)
        else:
            self._logger = None

    def path(self) -> str | None:
        """返回当前日志文件路径（便于测试/外部读取）。"""
        return self.log_file

    def send(self, event: dict) -> dict:
        line = json.dumps(event, ensure_ascii=False, default=str)
        self.buffer.append(line)
        if self._logger is not None:
            self._logger.info(line)
            self._handler.flush()  # 立即落盘，保证可读性与测试可观测性
        return {"channel": self.name, "ok": True, "detail": "logged"}

    def flush(self):
        if self._handler is not None:
            self._handler.flush()

    def close(self):
        if self._handler is not None:
            self._handler.flush()
            self._logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None


class WebhookChannel(Channel):
    """Webhook 渠道：HTTP POST JSON。依赖 requests，缺失则降级。"""

    name = "webhook"

    def __init__(self, url: str, timeout: float = 3.0, headers: dict | None = None):
        self.url = url
        self.timeout = timeout
        self.headers = headers or {"Content-Type": "application/json"}

    def healthy(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except Exception:
            return False

    def send(self, event: dict) -> dict:
        try:
            import requests
        except Exception:
            return {"channel": self.name, "ok": False,
                    "detail": "缺少 requests，无法推送（已降级）"}
        try:
            resp = requests.post(self.url, json=event,
                                 headers=self.headers, timeout=self.timeout)
            return {"channel": self.name, "ok": resp.status_code < 400,
                    "status": resp.status_code}
        except Exception as e:
            return {"channel": self.name, "ok": False, "detail": str(e)}


class SignalNotifier:
    """信号推送器：聚合多个渠道，对齐 BaseBroker 的调用风格。"""

    def __init__(self, channels: list[Channel] | None = None,
                 log_file: str | None = None):
        self.channels: list[Channel] = list(channels or [])
        if not self.channels and log_file is not None:
            self.channels.append(LogChannel(log_file))
        if not self.channels:
            self.channels.append(LogChannel())  # 默认仅日志
        self.history: list[dict] = []

    def attach(self, channel: Channel):
        self.channels.append(channel)

    def detach(self, name: str):
        self.channels = [c for c in self.channels if c.name != name]

    def push(self, event: dict) -> dict:
        """推送一个信号事件到所有渠道，返回聚合结果。"""
        event = dict(event)
        event.setdefault("ts", _dt.datetime.now().isoformat(timespec="seconds"))
        results = [ch.send(event) for ch in self.channels]
        summary = {
            "event": event,
            "delivered": sum(1 for r in results if r.get("ok")),
            "total": len(results),
            "results": results,
        }
        self.history.append(summary)
        return summary

    def notify_signal(self, symbol: str, action: str, price: float = 0.0,
                      strategy: str = "", **extra) -> dict:
        """便捷方法：推送一笔交易信号。action: BUY/SELL/HOLD 等。"""
        event = {"type": "signal", "symbol": symbol, "action": action,
                 "price": price, "strategy": strategy, **extra}
        return self.push(event)

    def last(self) -> dict | None:
        return self.history[-1] if self.history else None


# ===================== 第十四轮：微信推送（实盘结果推送） =====================
import urllib.request  # noqa: E402
import urllib.error    # noqa: E402


class WeChatNotifier:
    """企业微信群机器人 Webhook 推送（零依赖 urllib）。

    只需群机器人的 webhook key（或完整 URL）。只能推到该群，不能点对点。
    用法：WeChatNotifier("693a91f6-xxxx") 或 WeChatNotifier(完整URL)。
    """

    _BASE = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="

    def __init__(self, webhook_key: str, timeout: int = 10):
        self.key = webhook_key or ""
        self.timeout = timeout
        if "key=" in self.key:
            self.url = self.key if self.key.startswith("http") else (self._BASE + self.key.split("key=", 1)[1])
        else:
            self.url = self._BASE + self.key

    def push(self, text: str) -> bool:
        if not self.key:
            return False
        payload = json.dumps({"msgtype": "text",
                              "text": {"content": str(text)}}).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("errcode", -1) == 0
        except Exception as e:
            print("[WeChatNotifier] push failed: %s" % e)
            return False


class WeComAppNotifier:
    """企业微信自建应用推送（点对点推到个人，如黄子州）。

    凭证（corpid/secret/agentid/touser）从本地配置文件加载，**不进源码/不进 git**。
    配置文件 wecom_config.json 形如：
        {"corpid":"wwxxx","corpsecret":"xxx","agentid":1000002,"touser":"黄子州"}
    """

    _TOKEN_URL = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    _SEND_URL = "https://qyapi.weixin.qq.com/cgi-bin/message/send"

    def __init__(self, config: dict | str, timeout: int = 10):
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config = json.load(f)
        self.corpid = config.get("corpid", "")
        self.corpsecret = config.get("corpsecret", "")
        self.agentid = config.get("agentid", 0)
        self.touser = config.get("touser", "@all")
        self.timeout = timeout
        self._token: str | None = None

    def _get_token(self) -> str | None:
        if self._token:
            return self._token
        url = "%s?corpid=%s&corpsecret=%s" % (self._TOKEN_URL, self.corpid, self.corpsecret)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("errcode", -1) == 0:
                self._token = data.get("access_token")
                return self._token
            print("[WeComAppNotifier] token failed: %s" % data)
        except Exception as e:
            print("[WeComAppNotifier] token error: %s" % e)
        return None

    def push(self, text: str) -> bool:
        token = self._get_token()
        if not token:
            return False
        payload = json.dumps({
            "touser": self.touser,
            "msgtype": "text",
            "agentid": int(self.agentid),
            "text": {"content": str(text)},
        }).encode("utf-8")
        url = "%s?access_token=%s" % (self._SEND_URL, token)
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("errcode", -1) == 0
        except Exception as e:
            print("[WeComAppNotifier] push failed: %s" % e)
            return False


class NotifyHub:
    """多渠道聚合推送：企业微信 + 群机器人 + 任意 callable。"""

    def __init__(self, handlers=None):
        self.handlers = list(handlers or [])

    def add(self, h):
        self.handlers.append(h)

    def push(self, text):
        for h in self.handlers:
            try:
                if callable(h) and not hasattr(h, "push"):
                    h(text)
                elif hasattr(h, "push"):
                    h.push(text)
            except Exception as e:
                print("[NotifyHub] handler error: %s" % e)
        return True


def build_notifier(cfg: dict | None):
    """根据配置 dict 构建 NotifyHub（或返回 None）。

    cfg 形如：
      {"wechat_webhook": "群机器人key或完整URL",
       "wecom": "wecom_config.json" 或 {"corpid":..., "corpsecret":..., ...}}
    仅当配置了有效渠道时才返回 NotifyHub。
    """
    if not cfg:
        return None
    handlers = []
    if cfg.get("wechat_webhook"):
        handlers.append(WeChatNotifier(cfg["wechat_webhook"]))
    if cfg.get("wecom"):
        try:
            handlers.append(WeComAppNotifier(cfg["wecom"]))
        except Exception as e:
            print("[build_notifier] wecom init failed: %s" % e)
    return NotifyHub(handlers) if handlers else None
