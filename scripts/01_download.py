"""Phase 1: check (and optionally fetch) every raw dataset the project needs.

Default run is check-only: it reports each dataset as PRESENT or MISSING and
lists what the user must supply. Nothing is downloaded without a flag.

    python scripts/01_download.py                      # check only
    python scripts/01_download.py --generate-synthea   # needs Java 17+, ~200 MB jar
    python scripts/01_download.py --synthea-sample     # fallback: pre-built sample CSVs
    python scripts/01_download.py --kaggle             # needs a Kaggle API token

Standard library only, so it runs before requirements are installed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ONE_GB = 1024**3
MAX_RETRIES = 8
# The current Synthea release is compiled for Java 17 (class file 61); the build plan's
# "Java 11+" predates that.
MIN_JAVA = 17

SYNTHEA_JAR_URL = (
    "https://github.com/synthetichealth/synthea/releases/download/"
    "master-branch-latest/synthea-with-dependencies.jar"
)
SYNTHEA_SAMPLE_URL = (
    "https://synthetichealth.github.io/synthea-sample-data/downloads/"
    "latest/synthea_sample_data_csv_latest.zip"
)

# name -> folder, required files and the columns each must have (case-insensitive)
DATASETS = {
    "synthea": {
        "label": "Synthea clinical CSVs (build plan 3.1)",
        "dir": RAW / "synthea",
        "files": {
            "patients.csv": ["Id", "BIRTHDATE", "GENDER"],
            "encounters.csv": ["Id", "START", "PATIENT"],
            "conditions.csv": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
            "observations.csv": ["DATE", "PATIENT", "CODE", "VALUE", "UNITS"],
            "medications.csv": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
            "allergies.csv": ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"],
        },
        "sep": ",",
    },
    "cvd": {
        "label": "Cardiovascular Disease dataset (build plan 3.2)",
        "dir": RAW / "cvd",
        "files": {
            "cardio_train.csv": ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                                 "cholesterol", "gluc", "smoke", "alco", "active", "cardio"],
        },
        "sep": ";",
        "kaggle": "sulianova/cardiovascular-disease-dataset",
    },
    "recipes": {
        "label": "Food.com recipes (build plan 3.3)",
        "dir": RAW / "recipes",
        "files": {
            "RAW_recipes.csv": ["name", "minutes", "tags", "nutrition", "ingredients"],
        },
        "sep": ",",
        "kaggle": "shuyangli94/food-com-recipes-and-user-interactions",
    },
    "sleep": {
        "label": "Sleep Health and Lifestyle dataset (build plan 3.4)",
        "dir": RAW / "sleep",
        "files": {
            "Sleep_health_and_lifestyle_dataset.csv": ["Gender", "Age", "Sleep Duration",
                                                       "Stress Level", "BMI Category", "Daily Steps"],
        },
        "sep": ",",
        "kaggle": "uom190346a/sleep-health-and-lifestyle-dataset",
    },
}


# ---------------------------------------------------------------- checks

def find_file(folder: Path, name: str) -> Path | None:
    """Exact match first, then case-insensitive (Windows downloads vary)."""
    exact = folder / name
    if exact.is_file():
        return exact
    if folder.is_dir():
        for p in folder.iterdir():
            if p.is_file() and p.name.lower() == name.lower():
                return p
    return None


def read_header(path: Path, sep: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline().strip()
    return [c.strip().strip('"') for c in first.split(sep)]


def check_dataset(name: str) -> dict:
    spec = DATASETS[name]
    result = {"name": name, "label": spec["label"], "dir": spec["dir"], "files": [], "ok": True}
    for fname, cols in spec["files"].items():
        path = find_file(spec["dir"], fname)
        entry = {"file": fname, "status": "MISSING", "detail": ""}
        if path is None:
            result["ok"] = False
        elif path.stat().st_size == 0:
            entry.update(status="EMPTY", detail="file is 0 bytes")
            result["ok"] = False
        else:
            header = {c.lower() for c in read_header(path, spec["sep"])}
            missing = [c for c in cols if c.lower() not in header]
            size_mb = path.stat().st_size / 1e6
            if missing:
                entry.update(status="BAD HEADER", detail=f"missing columns: {missing}")
                result["ok"] = False
            else:
                size_txt = f"{size_mb:,.1f} MB" if size_mb >= 1 else f"{size_mb * 1000:,.0f} KB"
                entry.update(status="PRESENT", detail=size_txt)
        result["files"].append(entry)
    source = spec["dir"] / "SOURCE.json"
    result["source"] = json.loads(source.read_text()) if source.is_file() else None
    return result


def java_executable() -> str | None:
    """JAVA_HOME wins over PATH, so a newer JDK can be used without editing PATH."""
    home = os.environ.get("JAVA_HOME")
    if home:
        exe = Path(home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if exe.is_file():
            return str(exe)
    return shutil.which("java")


def java_major_version() -> int | None:
    java = java_executable()
    if not java:
        return None
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
    if not m:
        return None
    major, minor = int(m.group(1)), int(m.group(2) or 0)
    return minor if major == 1 else major  # "1.8.0" -> 8, "17.0.2" -> 17


def kaggle_token_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (cfg / "kaggle.json").is_file()


# ---------------------------------------------------------------- downloads

def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "lphrs/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        size = r.headers.get("Content-Length")
    return int(size) if size else None


def download(url: str, dest: Path, allow_large: bool) -> None:
    size = remote_size(url)  # also verifies the link before downloading
    if size is not None and size > ONE_GB and not allow_large:
        raise SystemExit(f"{url} is {size / ONE_GB:.1f} GB. Re-run with --yes to allow it.")
    print(f"  downloading {url}")
    print(f"  size: {size / 1e6:,.1f} MB" if size else "  size: unknown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Large GitHub downloads can be reset mid-stream; retry and resume with a Range request.
    for attempt in range(1, MAX_RETRIES + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        if size and done >= size:
            break
        headers = {"User-Agent": "lphrs/0.1"}
        if done:
            headers["Range"] = f"bytes={done}-"
            print(f"  resuming at {done / 1e6:,.1f} MB (attempt {attempt}/{MAX_RETRIES})")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if done and r.status == 206 else "wb"
                if mode == "wb":
                    done = 0  # server ignored the Range header: start over
                with open(tmp, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if size:
                            print(f"\r  {done / size:6.1%}", end="", flush=True)
            print()
            if size is None or done >= size:
                break
        except (ConnectionError, TimeoutError, urllib.error.URLError, http.client.HTTPException) as e:
            print(f"\n  connection problem: {e}")
            if attempt == MAX_RETRIES:
                raise SystemExit(f"  download failed after {MAX_RETRIES} attempts. "
                                 f"Re-run the same command to resume from {tmp.name}.")
            time.sleep(min(5 * attempt, 30))
    if size is not None and tmp.stat().st_size != size:
        raise SystemExit(f"  size mismatch for {tmp.name}: got {tmp.stat().st_size}, expected {size}. "
                         "Re-run to resume, or delete the .part file to start over.")
    tmp.replace(dest)


def write_source(folder: Path, **info) -> None:
    info["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SOURCE.json").write_text(json.dumps(info, indent=2))


def generate_synthea(patients: int, allow_large: bool) -> None:
    print("\n[synthea] generating locally")
    major = java_major_version()
    if major is None or major < MIN_JAVA:
        raise SystemExit(f"  Java {MIN_JAVA}+ is required; found {major or 'no Java'} at {java_executable()}. "
                         f"Install a JDK {MIN_JAVA}+ and point JAVA_HOME at it, or use --synthea-sample.")
    build = RAW / "synthea_build"
    jar = build / "synthea-with-dependencies.jar"
    if not jar.is_file():
        download(SYNTHEA_JAR_URL, jar, allow_large)
    cmd = [java_executable(), "-jar", jar.name, "-p", str(patients), "-s", str(SEED),
           "--exporter.csv.export=true", "--exporter.fhir.export=false",
           "--exporter.years_of_history=15"]
    print(f"  using Java {major}: {cmd[0]}")
    print("  running:", " ".join(cmd[1:]))
    if subprocess.run(cmd, cwd=build).returncode != 0:
        raise SystemExit("  Synthea failed (see its output above). Nothing was copied.")
    out = build / "output" / "csv"
    dest = DATASETS["synthea"]["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for csv in out.glob("*.csv"):
        shutil.copy2(csv, dest / csv.name)
        copied += 1
    print(f"  copied {copied} CSV files to {dest.relative_to(ROOT)}")
    write_source(dest, kind="generated", tool="Synthea", jar_url=SYNTHEA_JAR_URL,
                 java_major=major, args=cmd[3:])


def fetch_synthea_sample(allow_large: bool) -> None:
    print("\n[synthea] FALLBACK: using the pre-built sample, not locally generated data")
    zpath = RAW / "synthea_build" / "synthea_sample_data_csv_latest.zip"
    download(SYNTHEA_SAMPLE_URL, zpath, allow_large)
    dest = DATASETS["synthea"]["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        for m in members:
            (dest / Path(m).name).write_bytes(z.read(m))
    print(f"  extracted {len(members)} CSV files to {dest.relative_to(ROOT)}")
    write_source(dest, kind="fallback_sample", url=SYNTHEA_SAMPLE_URL)


def fetch_kaggle(name: str) -> None:
    spec = DATASETS[name]
    cli = shutil.which("kaggle")
    if not cli:
        raise SystemExit("  'kaggle' command not found. Run: pip install -r requirements.txt")
    spec["dir"].mkdir(parents=True, exist_ok=True)
    for fname in spec["files"]:
        print(f"\n[{name}] kaggle download {spec['kaggle']} -f {fname}")
        subprocess.run([cli, "datasets", "download", "-d", spec["kaggle"], "-f", fname,
                        "-p", str(spec["dir"])], check=True)
    for z in spec["dir"].glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(spec["dir"])
        z.unlink()
    write_source(spec["dir"], kind="kaggle_api", dataset=spec["kaggle"])


# ---------------------------------------------------------------- report

def print_report(results: list[dict], java: int | None, token: bool) -> list[str]:
    print("\n" + "=" * 72)
    print("LPHRS dataset status")
    print("=" * 72)
    todo = []
    for r in results:
        tag = "PRESENT" if r["ok"] else "MISSING"
        print(f"\n[{tag}] {r['name']}: {r['label']}")
        print(f"  folder: {r['dir'].relative_to(ROOT)}")
        if r["source"]:
            print(f"  source: {r['source'].get('kind')}")
            if r["source"].get("kind") == "fallback_sample":
                print("  NOTE: Synthea fallback sample in use (not the 1,000-patient generated set)")
        for f in r["files"]:
            print(f"    {f['status']:<10} {f['file']}  {f['detail']}")
        if not r["ok"]:
            todo.append(r["name"])

    print("\n" + "-" * 72)
    print(f"Java: {java if java else 'not found'} "
          f"({'OK for Synthea' if java and java >= MIN_JAVA else f'Synthea generation needs {MIN_JAVA}+'})")
    print(f"Kaggle API token: {'found' if token else 'not found'}")

    actions = []
    if "synthea" in todo:
        if java and java >= MIN_JAVA:
            actions.append("Synthea: run  python scripts/01_download.py --generate-synthea")
        else:
            actions.append(f"Synthea: install Java {MIN_JAVA}+ then run  python scripts/01_download.py "
                           "--generate-synthea  (or approve the fallback: --synthea-sample)")
    for name in ("cvd", "recipes", "sleep"):
        if name in todo:
            spec = DATASETS[name]
            files = ", ".join(spec["files"])
            actions.append(f"{name}: download {files} from "
                           f"https://www.kaggle.com/datasets/{spec['kaggle']} "
                           f"and place it in {spec['dir'].relative_to(ROOT)}{os.sep}")

    print("-" * 72)
    if actions:
        print("You must supply:")
        for i, a in enumerate(actions, 1):
            print(f"  {i}. {a}")
    else:
        print("All datasets present. Ready for Phase 2.")
    return todo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate-synthea", action="store_true", help="download Synthea jar and generate patients (Java 17+)")
    ap.add_argument("--patients", type=int, default=1000, help="Synthea population size (default 1000)")
    ap.add_argument("--synthea-sample", action="store_true", help="fallback: download pre-built Synthea sample CSVs")
    ap.add_argument("--kaggle", action="store_true", help="download missing Kaggle datasets with the Kaggle API")
    ap.add_argument("--yes", action="store_true", help="allow downloads larger than 1 GB")
    args = ap.parse_args()

    if args.generate_synthea and args.synthea_sample:
        ap.error("choose either --generate-synthea or --synthea-sample")

    if args.generate_synthea:
        generate_synthea(args.patients, args.yes)
    elif args.synthea_sample:
        fetch_synthea_sample(args.yes)

    if args.kaggle:
        if not kaggle_token_present():
            print("\nKaggle token not found (~/.kaggle/kaggle.json). Skipping Kaggle downloads.")
        else:
            for name in ("cvd", "recipes", "sleep"):
                if not check_dataset(name)["ok"]:
                    fetch_kaggle(name)

    results = [check_dataset(n) for n in DATASETS]
    todo = print_report(results, java_major_version(), kaggle_token_present())
    return 1 if todo else 0


if __name__ == "__main__":
    sys.exit(main())
