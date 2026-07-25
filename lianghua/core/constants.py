"""平台常量：交易日、配色、默认参数。"""
from __future__ import annotations

# 版本
VERSION = "1.6.0-live"

# 年度交易日（A 股近似）
TRADING_DAYS = 252

# A 股配色（红涨绿跌）
COLOR_UP = "#ff4d4f"
COLOR_DOWN = "#00d486"
COLOR_ACCENT = "#667eea"
ACCENT = COLOR_ACCENT  # UI 别名

# 默认回测参数
DEFAULT_INIT_CASH = 1_000_000.0
DEFAULT_COST_RATE = 0.0005

# 资产中文名
ASSET_CN = {
    "stock": "股票", "fund": "基金", "future": "期货", "option": "期权",
}

# 数据源超时（秒）
FETCH_TIMEOUT = 15
