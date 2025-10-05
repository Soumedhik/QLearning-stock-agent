"""Quick validator: byte-compile all .py files under src/ to catch syntax errors."""
import compileall
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

def main():
    if not SRC.exists():
        print(f"No src/ directory found at {SRC}")
        return
    print(f"Compiling Python files in {SRC}...")
    success = compileall.compile_dir(str(SRC), force=True, quiet=1)
    if success:
        print("All files compiled OK.")
    else:
        print("Some files failed to compile. See output above.")

if __name__ == '__main__':
    main()
