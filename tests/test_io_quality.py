"""IO 数据读写层回归：健壮错误处理 + 脏数据数值化 + validate_bars 数据质量校验。"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

import lianghua.io as D


def _tmp(suffix):
    fd, path = __import__("tempfile").mkstemp(suffix=suffix)
    os.close(fd)
    return path


def test_read_bars_missing_file_clear_error():
    try:
        D.read_bars(_tmp(".csv") + ".nope")
        raise AssertionError("缺失文件应抛 FileNotFoundError")
    except FileNotFoundError as e:
        assert "不存在" in str(e)


def test_read_bars_dirty_numeric_coerced():
    p = _tmp(".csv")
    # 故意放入非数值脏数据，应被安全数值化为 0 而不崩溃
    pd.DataFrame({"date": ["2023-01-01", "2023-01-02"],
                  "close": ["abc", "10.5"]}).to_csv(p, index=False)
    df = D.read_bars(p)
    assert df["close"].iloc[0] == 0.0 and df["close"].iloc[1] == 10.5
    os.remove(p)


def test_read_bars_missing_date_not_zero():
    p = _tmp(".csv")
    pd.DataFrame({"close": [10.0, 11.0]}).to_csv(p, index=False)
    df = D.read_bars(p)
    # date 缺失时应生成占位日期字符串，而非 0.0 数值
    assert df["date"].iloc[0] != 0 and str(df["date"].iloc[0]).startswith("1970")
    os.remove(p)


def test_validate_bars_detects_anomalies():
    bad = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-02"],
        "open": [10.0, 11.0, 9.0],
        "high": [10.5, 9.0, 10.0],   # 第2行 high<low
        "low": [9.5, 11.5, 8.5],
        "close": [-1.0, 10.5, 9.0],  # 第1行 close<=0
        "volume": [1000.0, 500.0, -3.0],  # 第3行 volume<0
    })
    issues = D.validate_bars(bad)
    assert any("high<low" in i for i in issues)
    assert any("close<=0" in i for i in issues)
    assert any("volume<0" in i for i in issues)
    assert any("重复日期" in i for i in issues)


def test_validate_bars_clean_passes():
    good = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "open": [10.0, 11.0], "high": [10.5, 11.5],
        "low": [9.5, 10.5], "close": [10.2, 11.2], "volume": [1000.0, 900.0],
    })
    assert D.validate_bars(good) == []


def test_roundtrip_preserved():
    p = _tmp(".csv")
    df = pd.DataFrame({"date": ["2023-01-02"], "open": [10], "high": [11],
                       "low": [9.8], "close": [10.6], "volume": [1000]})
    D.write_bars(df, p)
    assert D.read_bars(p).iloc[0]["close"] == 10.6
    tp = _tmp(".csv")
    D.write_trades([{"date": "2023-01-02", "side": "BUY", "price": 10.6}], tp)
    assert D.read_trades(tp)[0]["side"] == "BUY"
    os.remove(p)
    os.remove(tp)
