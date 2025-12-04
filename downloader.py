import os
import asyncio
import requests
from urllib.parse import quote

from utils import IGCSE_HOME

# --- CONFIGURATION ---

# 1. Define the subjects you are interested in.
# The keys are the subject codes. The values are tuples containing the
# exact "Level" and "Subject Name (Code)" folder names from the PapaCambridge URL.
# You may need to visit the website to confirm these folder names if you add new subjects.

SUBJECT_METADATA = {
    "9709": ("CAIE/CAIE-pastpapers/upload"),
    "9702": ("CAIE/CAIE-pastpapers/upload"),
    "9231": ("CAIE/CAIE-pastpapers/upload"),
    "0606": ("CAIE/CAIE-pastpapers/upload"),
    "0580": ("CAIE/CAIE-pastpapers/upload"),
    "0625": ("CAIE/CAIE-pastpapers/upload"),
}
PAPER_RANGE = range(1, 8)
VARIANT_RANGE = range(1, 4)
YEARS = [2025]  # range(2025, 2026)


PAPER_TYPES = ["qp"]

DOWNLOAD_DIR = IGCSE_HOME

# --- SCRIPT LOGIC (No need to edit below this line) ---

BASE_URL = "https://pastpapers.papacambridge.com/directories"

"""
https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9709_m24_qp_12.pdf
https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/0580_s25_qp_42.pdf
"""


async def download_file(url, local_path):
    """Downloads a file from a URL to a local path if it doesn't already exist."""
    local_dir = os.path.dirname(local_path)

    if os.path.exists(local_path):
        return

    os.makedirs(local_dir, exist_ok=True)

    try:
        with requests.Session() as s:
            print(f"Downloading File .... {url}")
            response = await asyncio.to_thread(s.get, url)

            if (
                response.status_code == 200
                and response.headers["content-type"] == "application/pdf"
            ):
                with open(local_path, "wb") as f:
                    f.write(response.content)
                print(f"[+] SUCCESS: Downloaded {local_path}")
            elif response.status_code == 404:

                print(f"[-] NOT FOUND: {os.path.basename(local_path)}")
                pass
            else:

                print(
                    f"[-] SKIPPING: {os.path.basename(local_path)} (does NOT exit)"
                )

    except requests.exceptions.RequestException as e:
        print(
            f"[!] NETWORK ERROR: Could not download {os.path.basename(local_path)}. Error: {e}"
        )


async def run_downloader():
    """Main function to iterate through all combinations and trigger downloads."""
    print("--- Starting Cambridge Past Paper Downloader ---")

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created base directory: {DOWNLOAD_DIR}")

    total_attempts = 0
    all_task = []
    for code, (path_prefix) in SUBJECT_METADATA.items():
        for year in YEARS:
            for session_code in ["s", "m", "w"]:
                year_short = str(year)[-2:]
                for paper_num in PAPER_RANGE:
                    for variant_num in VARIANT_RANGE:
                        paper_variant = f"{paper_num}{variant_num}"

                        for paper_type in PAPER_TYPES:
                            total_attempts += 1
                            filename = f"{code}_{session_code}{year_short}_{paper_type}_{paper_variant}.pdf"

                            url = f"{BASE_URL}/{path_prefix}/{filename}"
                            local_path = os.path.join(
                                DOWNLOAD_DIR,
                                code,
                                "exams",
                                filename,  # str(year),
                            )

                        all_task.append(
                            asyncio.create_task(download_file(url, local_path))
                        )

    await asyncio.gather(*all_task)
    print("\n--- Download process complete. ---")
    print(
        f"All specified years and subjects have been checked. Files are in '{DOWNLOAD_DIR}'."
    )


if __name__ == "__main__":
    asyncio.run(run_downloader())
