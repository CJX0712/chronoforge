"""ChronoForge CLI 入口（argparse）。

用法:
  python -m chronoforge.cli demo   # 跑端到端基准 + 落盘 benchmark.json
  python -m chronoforge.cli detcheck  # 同 seed 两次运行验确定性
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import determinism_check, run, save_report

_SEEDS = (7, 42, 123, 2024, 99)


def _print_table(result: dict) -> None:
    methods = result["methods"]
    print("\n=== ChronoForge 基准（sMAPE%, 越小越好）===")
    print(f"{'method':<20}{'sMAPE_mean':>14}{'sMAPE_std':>12}{'MASE_mean':>12}{'n':>6}")
    order = ["chronofuse", "average"] + [m for m in methods if m not in ("chronofuse", "average")]
    for m in order:
        if m not in methods:
            continue
        d = methods[m]
        print(
            f"{m:<20}{d['smape_mean']:>14.3f}{d['smape_std']:>12.3f}{d['mase_mean']:>12.3f}{d['n']:>6}"
        )
    print("\n=== 融合旗舰表现 ===")
    strongest = result["strongest_single_base"]
    strongest_smape = result["strongest_single_smape"]
    print(f"最强单基            : {strongest}  sMAPE={strongest_smape:.3f}")
    print(f"ChronoFuse sMAPE   : {result['chrono_smape']:.3f}")
    red = result["chrono_vs_strongest_smape_reduction_pct"]
    print(f"相对最强单基下降   : {red:+.2f}%  (门槛 >=5% 即达标 S 级)")
    print(f"非劣守卫回退次数  : {result['chrono_fallback_count']} / {result['n_series']}")
    print(f"滚动原点数        : {result.get('n_rolling_origins', '?')}")
    print(f"Tier-0 参考可用    : {result['reference_available']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chronoforge", description="ChronoForge 时间序列预测系统")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("demo", "detcheck"):
        p = sub.add_parser(c)
        p.add_argument("--horizon", type=int, default=12)
        p.add_argument("--regimes", type=str, default=None, help="逗号分隔 regime，默认全部")
        p.add_argument("--out", type=str, default="benchmark.json")
        p.add_argument("--noise", type=float, default=1.0)
        p.add_argument("--no-safeguard", action="store_true")
    args = ap.parse_args(argv)

    regimes = args.regimes.split(",") if args.regimes else None
    kwargs = dict(
        horizon=args.horizon,
        regimes=regimes,
        seeds=_SEEDS,
        noise_sigma=args.noise,
        safeguard=(not args.no_safeguard),
    )

    if args.cmd == "detcheck":
        dc = determinism_check(**kwargs)
        print("=== 确定性校验 ===")
        print(json.dumps(dc, ensure_ascii=False, indent=2))
        return 0 if dc["bit_identical"] else 1

    result, _ = run(**kwargs)
    save_report(result, args.out)
    _print_table(result)
    print(f"\n报告已落盘: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
