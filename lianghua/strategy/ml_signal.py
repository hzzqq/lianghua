"""ML 信号：逻辑回归 / 随机森林择时（sklearn 可选）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = c.pct_change()
    out["ret_5"] = c.pct_change(5)
    out["vol_20"] = c.pct_change().rolling(20).std()
    out["mom_20"] = c / c.rolling(20).mean() - 1
    return out


def ml_signal(df: pd.DataFrame, target_horizon: int = 5,
              model: str = "lr", test_size: float = 0.3,
              refit_step: int = 20, min_train: int = 60) -> pd.Series:
    """用未来 horizon 收益方向训练，以 walk-forward 方式预测当日方向（1 多 / -1 空）。

    严格因果（无前视），第 t 根 K 线的预测只依赖 t 及之前的数据：
    - 训练样本的标签窗口必须已闭合：特征行 i 的标签用到 close[i+H]，仅当 i+H <= t
      时才允许进入第 t 次训练（扩展窗口）；
    - 每 ``refit_step`` 根重拟合一次，两次拟合之间沿用上一个模型（预测仍只用旧模型）；
    - 可用训练样本不足 ``min_train`` 时退化为动量规则信号（滚动统计，本身无前视）。

    历史 bug（R19 修复）：旧实现用全样本 70/30 切分（``train_test_split`` 按全序列长度
    定训练边界），边界位置取决于"未来还有多少数据"——删掉尾部 K 线会移动边界、改变
    历史信号。被 ``test_reliability_audit::test_strategy_no_look_ahead[ml]`` 的未来扰动法
    抓出（删未来 100 根后前 200 个信号中 74 处改变 → 前视偏差，虚假收益源头之一）。
    ``test_size`` 参数保留仅为兼容旧调用，不再参与任何计算。

    sklearn 不可用时整体退化为动量与势头的简单规则信号。
    """
    feat = _features(df).dropna()
    c = df["close"].astype(float)
    fwd = c.pct_change(target_horizon).shift(-target_horizon)
    idx = feat.index
    n = len(feat)
    y = np.where(fwd.loc[idx] > 0, 1, -1)
    X = feat.values
    # feat 各行在 df 中的位置（升序）：用于判定"标签窗口在 t 时刻是否已闭合"
    pos = np.asarray(pd.Index(df.index).get_indexer(idx))
    pred = np.empty(n, dtype=int)
    min_tr = max(int(min_train), target_horizon + 2)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        have_sklearn = True
    except Exception:
        have_sklearn = False
    clf = None
    fitted_rows = -(10 ** 9)  # 上次拟合时的可用训练行数（驱动 refit 节奏）
    for p in range(n):
        # 可用作训练的行数：标签窗口已闭合（pos[i] + H <= pos[p]）的特征行数
        usable = int(np.searchsorted(pos, pos[p] - target_horizon, side="right"))
        if have_sklearn and usable >= min_tr and (
                clf is None or usable - fitted_rows >= refit_step):
            try:
                if model == "rf":
                    clf = RandomForestClassifier(n_estimators=50, random_state=0)
                else:
                    clf = LogisticRegression(max_iter=200)
                clf.fit(X[:usable], y[:usable])
                fitted_rows = usable
            except Exception:  # noqa: BLE001 - 拟合失败退回规则信号
                clf = None
        if clf is not None:
            pred[p] = int(clf.predict(X[p:p + 1])[0])
        else:
            # 退化：动量>0 多，否则空（滚动统计，无前视）
            pred[p] = 1 if feat["mom_20"].values[p] > 0 else -1
    return pd.Series(pred, index=idx, name="ml")
