#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""roam.py ─ Randomly drive the robot and capture data.

This script issues random movement commands, calls collect_data.main(),
and saves the command list to *commands.txt*.

If the user presses **Ctrl-C** we catch SIGINT in `signal_handler()` and
perform file-system hygiene:
    Remove orphaned images in ./data (e.g. *before_123.jpg* without
     matching *after_123.jpg*, or vice-versa).
    Remove any temporary *.tmp* files left by a partially executed
     run.
Then the program exits gracefully with status 0.
"""

import glob
import os
import random
import re
import signal
import sys

import collect_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BEFORE_GLOB = os.path.join(DATA_DIR, "before_*.jpg")
AFTER_GLOB = os.path.join(DATA_DIR, "after_*.jpg")
TEMP_GLOB = os.path.join(DATA_DIR, "*.tmp")


def signal_handler(sig, frame):
    """Clean up orphaned files and exit gracefully."""
    print("\n[CTRL-C] Cleaning up partial data ...")

    # Build quick lookup sets of file stems without the before_/after_ prefix
    before_files = glob.glob(BEFORE_GLOB)
    after_files = glob.glob(AFTER_GLOB)

    before_stems = {re.sub(r"^before_|\.jpg$", "", os.path.basename(p)) for p in before_files}
    after_stems = {re.sub(r"^after_|\.jpg$", "", os.path.basename(p)) for p in after_files}

    # Orphans: in one set but not the other
    orphan_before = before_stems - after_stems
    orphan_after = after_stems - before_stems

    removed = 0
    for stem in orphan_before:
        path = os.path.join(DATA_DIR, f"before_{stem}.jpg")
        try:
            os.remove(path)
            removed += 1
            print(f"  removed {path} (no matching after_*.jpg)")
        except OSError as exc:
            print(f"  WARN: could not remove {path}: {exc}")

    for stem in orphan_after:
        path = os.path.join(DATA_DIR, f"after_{stem}.jpg")
        try:
            os.remove(path)
            removed += 1
            print(f"  removed {path} (no matching before_*.jpg)")
        except OSError as exc:
            print(f"  WARN: could not remove {path}: {exc}")

    # Delete any lingering *.tmp files
    for tmp in glob.glob(TEMP_GLOB):
        try:
            os.remove(tmp)
            removed += 1
            print(f"  removed temporary file {tmp}")
        except OSError as exc:
            print(f"  WARN: could not remove {tmp}: {exc}")

    print(f"Cleanup done - {removed} extraneous file(s) deleted. Bye!\n")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

def random_commands(n):
    commands = ["forward", "backward", "clockwise", "counterclockwise"]
    distance_commands = ["forward", "backward"]
    result = ""
    prev_command = None
    for i in range(n):
        command = commands[random.randint(0, 3)]
        if command != prev_command:
            quantity = random.randint(10, 314) / 100 if command not in distance_commands else random.randint(1000, 10000) / 200
            result += f"{command} {quantity}\n"
            prev_command = command
    return result[:-1]


def main():
    while True:
        commands_number = random.randint(1, 3)
        commands = random_commands(commands_number)
        with open("commands.txt", "w") as command_file:
            command_file.write(commands)
        try:
            collect_data.main()
        except Exception as exc:
            print(f"ERROR, data collection failed: {exc}")
            sys.exit(1)

        


if __name__ == "__main__":
    main()
