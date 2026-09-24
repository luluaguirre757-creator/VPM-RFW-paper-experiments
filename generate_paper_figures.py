#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
VENV_PYTHON=(ROOT/".venv"/("Scripts/python.exe" if os.name=="nt" else "bin/python"))


def _same_executable(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except OSError:
        return False


# Keep this before every third-party import. VSCode's "Run Python File" and a
# PowerShell command may select a global Python whose compiled NumPy/pandas
# wheels are missing or ABI-incompatible. Re-run this same read-only command in
# the project environment rather than importing those broken global packages.
if VENV_PYTHON.is_file() and not _same_executable(Path(sys.executable),VENV_PYTHON):
    print(f"[paper-figures] switching to project environment: {VENV_PYTHON}",flush=True)
    completed=subprocess.run([str(VENV_PYTHON),str(Path(__file__).resolve()),*sys.argv[1:]],
                             cwd=str(ROOT),check=False)
    raise SystemExit(completed.returncode)

# Some managed Windows installations do not allow Matplotlib to write under
# AppData. Set this before importing pyplot so standalone generation is quiet
# and reliable without requiring user-level permissions.
MPL_CACHE=ROOT/"tmp"/"matplotlib"
MPL_CACHE.mkdir(parents=True,exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(MPL_CACHE))

from vpmrfw.plotting.paper_figures import generate_paper_figures


def _latest_results_dir() -> Path:
    candidates=[]
    for parent in (ROOT/"results", ROOT/"results_fast"):
        if not parent.is_dir():
            continue
        for path in parent.iterdir():
            source=path/"raw"/"synthetic_main.csv"
            if path.is_dir() and source.is_file():
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No existing run containing raw/synthetic_main.csv was found under "
            f"{ROOT/'results'} or {ROOT/'results_fast'}. Pass --results-dir explicitly."
        )
    return max(candidates,key=lambda p:(p.stat().st_mtime,p.name))


def main():
    parser=argparse.ArgumentParser(description="Regenerate paper figures/tables from an existing run; never reruns experiments.")
    parser.add_argument("--results-dir",type=Path,default=None,
                        help="Existing run directory (default: latest usable run under results/ or results_fast/)")
    parser.add_argument("--bootstrap-resamples",type=int,default=2000)
    args=parser.parse_args()
    results_dir=args.results_dir if args.results_dir is not None else _latest_results_dir()
    print(f"[paper-figures] source run: {results_dir.resolve()}")
    result=generate_paper_figures(results_dir,args.bootstrap_resamples)
    print(f"[paper-figures] generated {len(result['manifest'])} PDF figures in {result['output_dir']}")


if __name__ == "__main__":
    main()
