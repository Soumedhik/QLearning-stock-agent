"""Reorganise repository files into a structured layout.

This script moves known files into target folders. Run from the repository root:

    python .\scripts\restructure_repo.py

It is safe to run multiple times; it skips files that are already in the destination.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAPPINGS = {
    # notebooks
    "notebooks": [
        "reinforcement_learning_stock_trading.ipynb",
        "reinforcement_learning_stock_trading_final.ipynb",
        "rl-implementation.ipynb",
        "rl-implementationv2.ipynb",
        "rl-implementationv3ipynb.ipynb",
        "notebookac82ee67bb.ipynb",
        "notebookae7334f636.ipynb",
        "vis.ipynb",
    ],
    # scripts / code
    "src": [
        "rl_stock_trading.py",
        "visulisation.py",
        "create_notebook.py",
    ],
    # assets
    "assets": [
        "abstract.txt",
        "desc.txt",
        "todo.md",
        "skill_progression.png",
        "action_distributions_area.eps",
        "comparative_learning_curves.eps",
        "cumulative_cost_trajectories.eps",
        "skill_profile_radar.eps",
    ],
    # docs
    "docs": [
        "Dataset Selection.md",
        "rl-implementation.ipynb",
    ],
}


def ensure_dirs(root: Path):
    for d in ["src", "notebooks", "scripts", "data", "assets", "docs"]:
        (root / d).mkdir(exist_ok=True)


def move_file(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        print(f"Skipping (same path): {src}")
        return
    if not src.exists():
        print(f"Not found, skipping: {src}")
        return
    if dest.exists():
        print(f"Destination exists, skipping: {dest}")
        return
    print(f"Moving {src} -> {dest}")
    shutil.move(str(src), str(dest))


def main():
    print(f"Repository root: {ROOT}")
    ensure_dirs(ROOT)

    for folder, files in MAPPINGS.items():
        for f in files:
            src = ROOT / f
            dest = ROOT / folder / f
            move_file(src, dest)

    # Move remaining top-level .py, .ipynb files not already moved into src or notebooks
    for p in ROOT.iterdir():
        if p.is_file():
            if p.suffix == ".py" and p.name not in ("scripts",):
                move_file(p, ROOT / "src" / p.name)
            if p.suffix == ".ipynb":
                move_file(p, ROOT / "notebooks" / p.name)

    print("Reorganization complete. Review changes before committing.")


if __name__ == "__main__":
    main()
