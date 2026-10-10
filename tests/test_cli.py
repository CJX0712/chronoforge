"""CLI 冒烟测试。"""

from __future__ import annotations

import json

from chronoforge.cli import main


def test_cli_demo_smoke(tmp_path):
    out = tmp_path / "bench.json"
    rc = main(["demo", "--regimes", "trend_linear,seasonal_add", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "methods" in data and "chronofuse" in data["methods"]


def test_cli_detcheck_smoke(tmp_path):
    rc = main(["detcheck", "--regimes", "trend_linear", "--out", str(tmp_path / "b.json")])
    assert rc == 0
