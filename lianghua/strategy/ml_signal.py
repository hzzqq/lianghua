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
                model: str = "lr", test_size: float = 0.3) -> pd.Series:
    """用未来 horizon 收益方向训练，预测当日方向（1 多 / -1 空）。

    sklearn 不可用时退化为动量与势头的简单规则信号。
    """
    feat = _features(df).dropna()
    fwd = df["close"].astype(float).pct_change(target_horizon).shift(-target_horizon)
    y = np.where(fwd.loc[feat.index] > 0, 1, -1)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(
            feat.values, y, test_size=test_size, shuffle=False)
        if model == "rf":
            clf = RandomForestClassifier(n_estimators=50, random_state=0)
        else:
            clf = LogisticRegression(max_iter=200)
        clf.fit(Xtr, ytr)
        pred = clf.predict(feat.values)
    except Exception:
        # 退化：动量>0 多，否则空
        pred = np.where(feat["mom_20"].values > 0, 1, -1)
    return pd.Series(pred, index=feat.index, name="ml")
