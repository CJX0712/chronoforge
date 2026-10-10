"""端到端演示：生成合成数据 → 多算法基准 → 打印表 + 落盘 benchmark.json。

作者：晨星。纯 numpy + 可选 statsmodels，零外部权重下载，离线可跑。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronoforge.cli import _SEEDS, _print_table
from chronoforge.pipeline import run, save_report


def main() -> int:
    t0 = time.time()
    result, _ = run(seeds=_SEEDS, noise_sigma=1.0)
    elapsed = time.time() - t0

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark.json"
    )
    save_report(result, out)
    _print_table(result)
    print(f"\n端到端耗时: {elapsed:.2f}s (预算 <=60s)")
    print(f"报告已落盘: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
