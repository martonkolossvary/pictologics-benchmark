"""Build every manuscript figure and table from frozen outputs.

The command reads the compliance releases, the frozen workspace and any number
of benchmark run folders, and never runs an adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from bench.figures import style
from bench.figures.accuracy import build_accuracy_figures
from bench.figures.checks import accessibility_views
from bench.figures.design import (
    build_design_figure,
    build_mask_supplement,
    build_noise_supplement,
)
from bench.figures.performance import build_performance_figures

RELEASES = {
    "ibsi1": "ibsi1",
    "phase1": "ibsi2-phase1",
    "phase2": "ibsi2-phase2",
}


def native_outputs_table(contract: Path, output_dir: Path) -> None:
    """Table S2: native outputs per package and group; runtime is never divided by them."""

    payload = json.loads(contract.read_text(encoding="utf-8"))
    rows = [
        {
            "package": style.ADAPTER[key].label,
            **{
                style.GROUP_LABEL[group]: count
                for group, count in inventory["family_output_counts"].items()
            },
        }
        for key, inventory in payload["adapter_inventories"].items()
    ]
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        tables / "tableS2_native_outputs_per_group.csv", index=False
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compliance-root", default="results/compliance")
    parser.add_argument("--workspace-root", default="data/benchmark")
    parser.add_argument("--contract", default="configs/benchmark/main.json")
    parser.add_argument(
        "--runs", nargs="*", default=[], help="Benchmark run folders (one per pillar)"
    )
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=128,
        help="Grid of the Figure 5 synthetic rows",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    root = Path(args.compliance_root)
    workspace = Path(args.workspace_root)
    reports = [root / RELEASES[key] / "report" for key in ("ibsi1", "phase1", "phase2")]
    records = []
    if all(report.is_dir() for report in reports):
        records += build_accuracy_figures(*reports, output)
    else:
        print(
            "The conformance results are incomplete; the accuracy figures are skipped."
        )
    records += [
        build_design_figure(workspace, output),
        build_mask_supplement(workspace, output),
        build_noise_supplement(workspace, output),
    ]
    if args.runs:
        records += build_performance_figures(
            [Path(run) for run in args.runs], output, synthetic_size=args.synthetic_size
        )
    native_outputs_table(Path(args.contract), output)
    for record in records:
        accessibility_views(
            output / f"{record['figure_id']}.png", output / "accessibility"
        )
    (output / "figures_manifest.json").write_text(
        json.dumps({"figures": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{len(records)} figures written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
