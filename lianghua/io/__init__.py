"""数据导入导出（迭代29）。

CSV 为核心能力（无额外依赖）；Excel 为可选（需 openpyxl / xlsxwriter）。
所有 Excel 路径均做优雅降级：缺依赖时抛出明确可读的错误而非崩溃。

IO 层契约：
- read/write_bars  : OHLCV 行情（date/open/high/low/close/volume）
- read/write_trades: 交易记录 list[dict] <-> CSV
- validate_bars    : 数据质量校验（脏数据/负价/高低错位等），防止垃圾进模型
"""
from __future__ import annotations

import os
import pandas as pd

__all__ = ["read_bars", "write_bars", "read_trades", "write_trades",
           "validate_bars", "to_excel", "read_excel_sheet", "excel_available"]

BARS_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_NUMERIC = ["open", "high", "low", "close", "volume"]


def _coerce_bars(df: pd.DataFrame) -> pd.DataFrame:
    """统一化为标准 OHLCV 结构。

    - 数值列缺失时补 0.0，且对非数值脏数据做安全数值化（不再抛出）；
    - date 缺失时生成占位日期序列（而非被错填成无意义的 0.0）。
    """
    df = df.copy()
    for c in _NUMERIC:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "date" not in df.columns:
        # 生成占位日期序列，避免 date 列被填成 0.0 污染下游
        df["date"] = pd.date_range(
            "1970-01-01", periods=len(df), freq="D"
        ).astype(str)
    return df[["date"] + _NUMERIC]


def validate_bars(df: pd.DataFrame) -> list[str]:
    """校验 OHLCV 数据质量，返回问题列表（空列表表示通过）。

    检测：缺失/非数值、非正价格、最高<最低、负成交量、空数据。
    """
    issues: list[str] = []
    if df is None or len(df) == 0:
        return ["行情为空"]
    df = _coerce_bars(df)
    if (df["close"] <= 0).any():
        issues.append("存在非正收盘价(close<=0)")
    if (df["open"] <= 0).any():
        issues.append("存在非正开盘价(open<=0)")
    if (df["high"] < df["low"]).any():
        issues.append("存在最高价低于最低价(high<low)的异常 K 线")
    if (df["volume"] < 0).any():
        issues.append("存在负成交量(volume<0)")
    if df["date"].duplicated().any():
        issues.append("存在重复日期")
    return issues


def read_bars(path: str) -> pd.DataFrame:
    """读取 OHLCV 行情 CSV（文件缺失时给出清晰报错）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"行情文件不存在: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"读取行情 CSV 失败: {path} -> {e}") from e
    return _coerce_bars(df)


def write_bars(df: pd.DataFrame, path: str) -> str:
    """写出 OHLCV 行情 CSV。"""
    out = _coerce_bars(df)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return path


def read_trades(path: str) -> list[dict]:
    """读取交易记录 CSV -> list[dict]（文件缺失时给出清晰报错）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"交易记录文件不存在: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"读取交易 CSV 失败: {path} -> {e}") from e
    return df.to_dict(orient="records")


def write_trades(trades: list[dict], path: str) -> str:
    """写出交易记录 list[dict] -> CSV。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pd.DataFrame(trades or []).to_csv(path, index=False)
    return path


def excel_available() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except Exception:
        try:
            import xlsxwriter  # noqa: F401
            return True
        except Exception:
            return False


def to_excel(sheets: dict, path: str) -> str:
    """将多个 DataFrame 写入一个 Excel 文件。

    sheets: {sheet_name: DataFrame}。
    需要 openpyxl 或 xlsxwriter，否则抛出可读错误。
    """
    if not excel_available():
        raise RuntimeError("导出 Excel 需要 openpyxl 或 xlsxwriter，请先 pip install openpyxl")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl" if _has("openpyxl") else "xlsxwriter") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=str(name)[:31], index=False)
    return path


def read_excel_sheet(path: str, sheet=0) -> pd.DataFrame:
    """读取 Excel 某一工作表。需要 openpyxl / xlsxwriter。"""
    if not excel_available():
        raise RuntimeError("读取 Excel 需要 openpyxl，请先 pip install openpyxl")
    return pd.read_excel(path, sheet_name=sheet)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False
