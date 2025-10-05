# rl-stock-trading

Reinforcement learning for stock trading in data-scarce environments — Q-learning agent, environment simulator, notebooks and reproducible scripts.

Summary of what I added

- Created folder structure: `src/`, `notebooks/`, `assets/`, `docs/`, `scripts/`, `data/` (empty)
- Added `scripts/restructure_repo.py` to move existing top-level files into the new folders
- Added `scripts/validate_src.py` to byte-compile `src/` and catch syntax errors
- Added `requirements.txt`, `.gitignore`, `CONTRIBUTING.md`, and `LICENSE` placeholder

Repository structure

- `src/`       - Python modules and scripts (production code)
- `notebooks/` - Jupyter notebooks and experiments
- `assets/`    - images, EPS figures and supplementary files
- `docs/`      - design notes and documentation
- `scripts/`   - maintenance scripts (restructuring, validation)
- `data/`      - datasets (ignored by .gitignore; use git-lfs or external storage for large data)

Quick start (PowerShell)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. (If not already run) reorganize the files into the new layout:

```powershell
python .\scripts\restructure_repo.py
```

3. Validate Python source files in `src`:

```powershell
python .\scripts\validate_src.py
```

4. Commit and push to GitHub:

```powershell
git init
git add .
git commit -m "chore: reorganize repository into structured layout"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Notes and recommendations

- Choose and add a proper license (replace `LICENSE` placeholder). Common choices: MIT, Apache-2.0.
- For large datasets, avoid committing raw data into `data/`; use git-lfs or external storage with links in `docs/`.
- Review and annotate each notebook in `notebooks/` with a short description cell explaining purpose and dependencies.
- If you prefer copying files instead of moving them, change `shutil.move` to `shutil.copy2` in `scripts/restructure_repo.py`.

Next steps I can help with

- Extract reusable functions from notebooks into `src/` and add unit tests
- Add CI (GitHub Actions) to run `scripts/validate_src.py` and run notebooks with `nbconvert`
- Add a recommended license and prepare a release-ready README with badges
# Reinforcement Learning - Stock Trading (Reorganized)

This repository contains experiments, notebooks and scripts for reinforcement learning applied to stock trading.

This commit adds a structured layout and helper script to reorganise the existing files into a clean codebase suitable for publishing on GitHub.

Top-level layout after running the reorganization script:

- src/           : Python modules and reusable code
- notebooks/     : Jupyter notebooks and exploratory work
- scripts/       : helper scripts (including the reorganization script)
- data/          : datasets (kept out of source control or referenced)
- assets/        : figures, plots, images, eps files
- docs/          : project documentation
- .gitignore     : sensible defaults for Python & notebooks
- requirements.txt: minimal environment to run notebooks and scripts

How to use
1. (Optional) Inspect `scripts/restructure_repo.py` to confirm where files will be moved.
2. Create and activate a Python environment and install dependencies:

   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

3. Run the restructure script from the repository root (PowerShell):

   python .\scripts\restructure_repo.py

The script will move known files into the new folders. It prints actions performed and skips files it doesn't find.

After running the script, verify the tree, review notebooks and then commit and push to GitHub.

If you want the script to actually copy instead of move, open the script and change `shutil.move` to `shutil.copy2` for the entries you want to preserve.

Repository structure (after running the script):

- `src/` - Python modules and scripts
- `notebooks/` - Jupyter notebooks and experiments
- `assets/` - figures, images, and supplementary files
- `docs/` - design docs and notes
- `scripts/` - maintenance scripts (restructure, checks)

Verification
- Run `python .\scripts\validate_src.py` to byte-compile Python source files and spot syntax errors.


Contact
If you'd like a different layout or for me to run the moves here, tell me and I can update the script or perform the reorganization.
