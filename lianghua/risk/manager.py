"""风控模块：仓位上限、止损、回撤熔断、黑名单，并按资产类别差异化。

与回测引擎解耦：引擎在撮合前调用 check_entry / check_exit，
在买入前调用 position_size 决定仓位。

v2 改进：
- check_exit 现按资产类别使用差异化止损线（此前只用全局默认值，是隐性 bug）。
- limits_for(None) 安全回退到股票默认，避免 None 资产类型触发 KeyError。
- 新增 max_position_for / drawdown_for 访问器与 summary() 可观测快照。
- position_size 现返回受单标的上限约束的可部署现金比例。
"""
from __future__ import annotations

from ..core.assets import AssetType


# 各资产类别默认风控参数
DEFAULT_LIMITS = {
    AssetType.STOCK:  dict(max_position=0.95, stop_loss=0.10, drawdown=0.25),
    AssetType.FUND:   dict(max_position=0.98, stop_loss=0.15, drawdown=0.30),
    AssetType.FUTURE: dict(max_position=0.90, stop_loss=0.08, drawdown=0.30, max_leverage=5.0, forced_liq=0.80),
    AssetType.OPTION: dict(max_position=0.50, stop_loss=0.50, drawdown=0.50, max_premium_frac=0.20),
}


class RiskManager:
    """策略的"安全阀"，支持按资产类别差异化配置。"""

    def __init__(
        self,
        max_position: float = 0.95,
        stop_loss: float = 0.10,
        max_drawdown: float = 0.25,
        blacklist: list[str] | None = None,
        limits: dict[AssetType, dict] | None = None,
    ):
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.max_drawdown = max_drawdown
        self.blacklist = set(blacklist or [])
        # 深拷贝默认参数，避免跨实例共享可变字典
        self.limits = {at: dict(DEFAULT_LIMITS[at]) for at in DEFAULT_LIMITS}
        # 用实例级全局默认值覆盖股票项，保证向后兼容（构造参数对 STOCK 生效）
        self.limits[AssetType.STOCK].update({
            "max_position": self.max_position,
            "stop_loss": self.stop_loss,
            "drawdown": self.max_drawdown,
        })
        if limits:
            for k, v in limits.items():
                self.limits[k] = {**self.limits.get(k, {}), **v}

    # ---------- 通用接口（向后兼容） ----------
    def check_entry(self, symbol: str, cash: float, asset_type: AssetType | None = None) -> bool:
        if symbol in self.blacklist:
            return False
        if cash <= 0:
            return False
        if asset_type and asset_type == AssetType.OPTION:
            # 期权需保留足够现金覆盖潜在权利金
            return cash > 0
        return True

    def check_exit(self, entry_price: float, current_price: float,
                   asset_type: AssetType | None = None) -> bool:
        """触发止损平仓。按资产类别使用差异化止损线。"""
        if entry_price <= 0:
            return False
        sl = self.stop_loss_for(asset_type)
        return (entry_price - current_price) / entry_price >= sl

    def position_size(self, cash: float, price: float,
                      asset_type: AssetType | None = None) -> float:
        """返回受单标的仓位上限约束的可部署现金比例（0~1）。"""
        if cash <= 0 or price <= 0:
            return 0.0
        return self.max_position_for(asset_type)

    # ---------- 资产差异化接口 ----------
    def limits_for(self, asset_type: AssetType | None) -> dict:
        if asset_type is None:
            return self.limits.get(AssetType.STOCK, DEFAULT_LIMITS[AssetType.STOCK])
        return self.limits.get(asset_type, self.limits[AssetType.STOCK])

    def max_position_for(self, asset_type: AssetType | None) -> float:
        return self.limits_for(asset_type).get("max_position", self.max_position)

    def max_leverage(self, asset_type: AssetType) -> float:
        return self.limits_for(asset_type).get("max_leverage", 1.0)

    def max_premium_frac(self, asset_type: AssetType) -> float:
        return self.limits_for(asset_type).get("max_premium_frac", 0.2)

    def stop_loss_for(self, asset_type: AssetType | None) -> float:
        return self.limits_for(asset_type).get("stop_loss", self.stop_loss)

    def drawdown_for(self, asset_type: AssetType | None) -> float:
        return self.limits_for(asset_type).get("drawdown", self.max_drawdown)

    def summary(self) -> dict:
        """返回当前风控配置的可观测快照，便于审计/日志。"""
        return {
            "blacklist": sorted(self.blacklist),
            "limits": {at.value: dict(self.limits[at]) for at in self.limits},
        }

    def check_margin(self, used_margin: float, equity: float, asset_type: AssetType) -> bool:
        """期货：保证金占用是否超过强平线（默认权益的 80%）。"""
        if asset_type != AssetType.FUTURE:
            return True
        if equity <= 0:
            return used_margin <= 0
        forced = self.limits_for(asset_type).get("forced_liq", 0.80)
        return used_margin <= equity * forced

    def premium_ok(self, premium: float, cash: float, asset_type: AssetType) -> bool:
        """期权：单笔权利金是否超占资上限。"""
        if asset_type != AssetType.OPTION:
            return True
        if premium < 0:
            return False
        cap = self.max_premium_frac(asset_type)
        return premium <= cash * cap
