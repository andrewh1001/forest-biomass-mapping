"""
download_appeears.py
--------------------
Bulk-download all files from a completed AppEEARS task without manual clicking.

Usage:
    python download_appeears.py

Set TASK_ID and OUT_DIR below, then run. You'll be prompted for your
NASA Earthdata username and password (never stored).
"""

import os
import getpass
import requests
from pathlib import Path
from tqdm import tqdm   # pip install tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TASK_ID = "YOUR_APPEEARS_TASK_ID"   # paste UUID from AppEEARS URL
OUT_DIR = "data/modis_lai"      # where to save files
# ---------------------------------------------------------------------------

BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"


def login(username, password):
    r = requests.post(f"{BASE_URL}/login", auth=(username, password))
    r.raise_for_status()
    return r.json()["token"]


def list_files(token, task_id):
    r = requests.get(
        f"{BASE_URL}/bundle/{task_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    r.raise_for_status()
    return r.json()["files"]   # list of {file_id, file_name, file_size, ...}


def download_file(token, task_id, file_id, file_name, out_dir):
    out_path = Path(out_dir) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  skip (exists): {file_name}")
        return

    r = requests.get(
        f"{BASE_URL}/bundle/{task_id}/{file_id}",
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        allow_redirects=True
    )
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        desc=file_name[-50:],
        total=total,
        unit="B", unit_scale=True, unit_divisor=1024,
        leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    print("AppEEARS Bulk Downloader")
    print("=" * 40)

    username = input("NASA Earthdata username: ")
    password = getpass.getpass("NASA Earthdata password: ")

    print("Logging in...")
    token = login(username, password)
    print("Login successful.")

    print(f"Fetching file list for task {TASK_ID}...")
    files = list_files(token, TASK_ID)
    print(f"Found {len(files)} files.")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f['file_name']}")
        download_file(token, TASK_ID, f["file_id"], f["file_name"], OUT_DIR)

    print(f"\nDone. Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()