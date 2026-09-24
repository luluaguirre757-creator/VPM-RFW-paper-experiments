#!/usr/bin/env python3
"""Run exactly the experiment families reported in the final paper."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
COMPONENTS = ("synthetic", "synthetic-budget", "synthetic-drift", "hsi", "convergence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test; not paper statistics")
    parser.add_argument("--component", choices=(*COMPONENTS, "certified-transfer"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    output = (args.output or ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = (args.component,) if args.component else (*COMPONENTS, "certified-transfer")

    for component in selected:
        if component == "certified-transfer":
            cmd = [
                sys.executable, "-m", "vpmrfw.experiments.synthetic_certified_bridge",
                "--out", str(output / component),
            ]
            if args.quick:
                cmd += ["--sequences", "1", "--tasks", "12", "--bootstrap", "100"]
        else:
            cmd = [
                sys.executable, str(ROOT / "run_experiments.py"),
                "--suite", component,
                "--config", str(ROOT / "configs" / "paper.yaml"),
                "--data-dir", str(args.data_dir),
                "--output", str(output / component),
                "--no-archive",
            ]
            if args.quick:
                cmd.append("--quick")
        print(f"[RUN_PAPER] {component}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    if args.component is None:
        redraw = [
            sys.executable, str(ROOT / "redraw_paper_figures.py"),
            "--results-dir", str(output),
            "--output-dir", str(output / "paper_figures"),
            "--bootstrap-resamples", "200" if args.quick else "2000",
        ]
        if args.quick:
            redraw += ["--allow-partial", "--allow-font-fallback"]
        print("[RUN_PAPER] final paper figures", flush=True)
        subprocess.run(redraw, cwd=ROOT, check=True)

        verify = [
            sys.executable, str(ROOT / "verify_reproduction.py"),
            "--results-dir", str(output),
        ]
        if args.quick:
            verify.append("--quick")
        print("[RUN_PAPER] reproduction verification", flush=True)
        subprocess.run(verify, cwd=ROOT, check=True)

    print(f"[done] {output}")


if __name__ == "__main__":
    main()
