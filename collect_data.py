#!/usr/bin/env python3
"""
collect_data.py
Sequential helper script for a ROSMaster-style data-collection workflow.

Changes in this version
-----------------------
• Images already go to data/<index>_{before|after}.jpg
• A copy of commands.txt (e.g. 00004.txt) is now written *both*
  - to the project root   -> 00004.txt
  - to the data directory -> data/00004.txt
"""

import subprocess
from pathlib import Path
import shutil
import sys

# ---------- Configuration ----------------------------------------------------
RAW_IMAGE   = Path("output.jpg")   # Filename emitted by take_image.py
NUMBER_WIDTH = 5                   # 00004 -> width 5
DATA_DIR    = Path("data")         # Where images & txt copy are stored
PYTHON_EXE  = sys.executable       # Usually "python3"
# -----------------------------------------------------------------------------


def next_index(workdir: Path, width: int = NUMBER_WIDTH) -> int:
    """Return the next integer index based on existing numbered files."""
    max_seen = -1
    for search_dir in (workdir, workdir / DATA_DIR):
        if not search_dir.exists():
            continue
        for p in search_dir.iterdir():
            stem = p.stem
            if len(stem) >= width and stem[:width].isdigit():
                max_seen = max(max_seen, int(stem[:width]))
    return max_seen + 1


def run_script(script: str) -> None:
    """Run another Python script and raise on failure."""
    print(f"-> Running {script} …")
    subprocess.run([PYTHON_EXE, script], check=True)
    print(f"Finished {script}")


def rename_and_move_image(src: Path, dst_stem: str, data_dir: Path) -> Path:
    """Move/rename `src` image into data_dir/<dst_stem>.<ext>."""
    if not src.exists():
        raise FileNotFoundError(f"Expected image {src} not found.")
    data_dir.mkdir(exist_ok=True)
    dst = data_dir / f"{dst_stem}{src.suffix}"
    src.rename(dst)
    print(f"{src.name} -> {dst}")
    return dst


def main() -> None:
    workdir = Path.cwd()
    idx = next_index(workdir)
    idx_str = f"{idx:0{NUMBER_WIDTH}d}"
    print(f"Starting collection cycle #{idx_str}")

    # 1. First image ("before")
    run_script("take_image.py")
    rename_and_move_image(RAW_IMAGE, f"{idx_str}_before", DATA_DIR)

    # 2. Run commands
    run_script("run_commands.py")

    # 3. Second image ("after")
    run_script("take_image.py")
    rename_and_move_image(RAW_IMAGE , f"{idx_str}_after", DATA_DIR)

    # 4. Archive commands.txt — root + data/
    src_cmds = workdir / "commands.txt"
    if not src_cmds.exists():
        raise FileNotFoundError("commands.txt not found.")

    # Data-dir copy
    DATA_DIR.mkdir(exist_ok=True)
    data_copy = DATA_DIR / f"{idx_str}.txt"
    shutil.copy2(src_cmds, data_copy)
    print(f"Copied commands.txt -> {data_copy}")

    print("Data-collection sequence complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
