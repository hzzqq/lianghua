"""risk/stress 单元测试：守卫 + 内置情景 + 汇总报告。"""
import numpy as np
import pandas as pd
import pytest

from lianghua.risk.stress import (
    BUILTIN_SCENARIOS,
    list_builtin_scenarios,
    stress_report,
    stress_test,
)


def _weights():
    return {"股票": 0.6, "债券": 0.3, "现金": 0.1}


def test_stress_test_basic():
    df = stress_test(_weights(), {"跌10%": {"股票": -0.1, "债券": 0.0, "现金": 0.0}})
    assert df.loc[0, "组合损益%"] == pytest.approx(-6.0)   # 0.6 * -0.1 * 100


def test_stress_test_missing_asset_treated_zero():
    df = stress_test(_weights(), {"仅股票跌": {"股票": -0.2}})
    assert df.loc[0, "组合损益%"] == pytest.approx(-12.0)


def test_stress_test_base_value():
    df = stress_test(_weights(), {"跌": {"股票": -0.1}}, base_value=1_000_000)
    assert "组合损益金额" in df.columns
    assert df.loc[0, "组合损益金额"] == pytest.approx(-60_000.0)


def test_stress_test_rejects_non_dict_weights():
    with pytest.raises(TypeError):
        stress_test([0.5, 0.5], {"x": {"股票": -0.1}})


def test_stress_test_rejects_nonfinite_weight():
    with pytest.raises(ValueError):
        stress_test({"股票": float("nan")}, {"x": {"股票": -0.1}})


def test_stress_test_rejects_nonfinite_shock():
    with pytest.raises(ValueError):
        stress_test(_weights(), {"x": {"股票": float("inf")}})


def test_builtin_scenario_by_name():
    df = stress_test(_weights(), {"2008": "2008_GFC"})
    assert df.loc[0, "情景"] == "2008"
    assert df.loc[0, "组合损益%"] < 0
    assert "2008_GFC" in BUILTIN_SCENARIOS


def test_unknown_builtin_raises():
    with pytest.raises(KeyError):
        stress_test(_weights(), {"x": "不存在"})


def test_base_value_invalid():
    with pytest.raises(ValueError):
        stress_test(_weights(), {"x": {"股票": -0.1}}, base_value=-5)


def test_stress_report_worst():
    rep = stress_report(_weights(), {"轻": {"股票": -0.1}, "重": {"股票": -0.3}})
    assert rep["worst_scenario"] == "重"
    assert rep["worst_pnl_pct"] == pytest.approx(-18.0)


def test_list_builtin_scenarios():
    names = list_builtin_scenarios()
    assert "2008_GFC" in names and "COVID_2020" in names
