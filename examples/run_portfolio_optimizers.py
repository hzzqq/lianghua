"""示例：遍历统一优化器注册表中的全部权重生成器并对比。

构造一个合成多资产日收益矩阵，分别用每个已注册优化器求解权重，
打印各优化器的权重向量与集中度（HHI），直观对比风险平价 / 最小方差 / HRP 等。

用法：
    python examples/run_portfolio_optimizers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lianghua.portfolio.registry import OPTIMIZER_NAMES, get_optimizer


def main(n_assets: int = 5, n_days: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = [f"A{i}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0.0005, 0.012, (n_days, n_assets)), columns=cols)
    print(f"合成收益矩阵：{n_assets} 资产 × {n_days} 日\n")
    rows = []
    for name in OPTIMIZER_NAMES:
        try:
            w = get_optimizer(name)(rets)
            w = w.reindex(cols).fillna(0.0)
            hhi = float((w ** 2).sum())
            rows.append({
                "优化器": name,
                "权重和": round(float(w.sum()), 3),
                "集中度HHI": round(hhi, 3),
                "最大权重": round(float(w.abs().max()), 3),
                "权重": ", ".join(f"{c}={v:.2f}" for c, v in w.items()),
            })
        except Exception as e:
            rows.append({"优化器": name, "权重和": None, "集中度HHI": None,
                         "最大权重": None, "权重": f"ERR:{e}"[:30]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n_assets=n)
