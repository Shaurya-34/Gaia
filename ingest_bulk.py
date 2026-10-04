"""Pull Gaia DR3 gaia_source from the bulk CDN, one file at a time.

Each GaiaSource_<lo>-<hi>.csv.gz holds the stars whose level-8 nested ICRS
HEALPix index (source_id >> 43) is in [lo, hi] (checked on real files). For each
file: download, verify the MD5 from _MD5SUM.txt, stream-filter with the PREREG
section 2 cuts (the same ones query_gaia.py sends to the archive), drop
sealed-block stars in memory, apply the zero point, write parquet with the same
columns as query_gaia.py, then delete the raw file. Resume-safe.

Run:  python ingest_bulk.py --block 6 --dry-run      (list files and total size)
      python ingest_bulk.py --block 6 --limit 1      (first file only)
"""
import argparse
import gzip
import hashlib
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from query_gaia import (COLUMNS, CUTS, SPLITS_PATH, apply_zero_point, drop_sealed,
                        icrs_pixels_for_block)

CDN = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/"
LISTING = "https://gaia.eu-1.cdn77-storage.com/"
PREFIX = "Gaia/gdr3/gaia_source/"
FILE_LEVEL = 8      # HEALPix level of the ranges in the file names
CHUNK_LEVEL = 5     # level used to map a block to files (64 level-8 pixels each)
READ_COLUMNS = [c.strip() for c in COLUMNS.split(",")] + ["parallax_over_error"]


def file_range(name):
    lo, hi = name[len("GaiaSource_"):-len(".csv.gz")].split("-")
    return int(lo), int(hi)


def list_files():
    """{name: {"md5": ..., "bytes": ...}} for every bulk file."""
    files = {}
    for line in requests.get(CDN + "_MD5SUM.txt", timeout=120).text.splitlines():
        if "GaiaSource_" in line:
            md5, name = line.split()
            files[name] = {"md5": md5}
    token = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(requests.get(LISTING, params=params, timeout=120).content)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", ns):
            name = c.find("s:Key", ns).text.rsplit("/", 1)[-1]
            if name in files:
                files[name]["bytes"] = int(c.find("s:Size", ns).text)
        if root.find("s:IsTruncated", ns).text != "true":
            return files
        token = root.find("s:NextContinuationToken", ns).text


def files_for_block(names, block):
    """Files whose level-8 range overlaps the level-5 chunks covering `block`
    (query_gaia.icrs_pixels_for_block, which includes a safety ring)."""
    chunks = np.array(icrs_pixels_for_block(block, CHUNK_LEVEL))
    scale = 4 ** (FILE_LEVEL - CHUNK_LEVEL)
    starts, ends = chunks * scale, chunks * scale + scale - 1
    keep = []
    for name in names:
        lo, hi = file_range(name)
        if np.any((starts <= hi) & (ends >= lo)):
            keep.append(name)
    return sorted(keep)


def apply_cuts(df):
    """The archive-side cuts of query_gaia.build_query, applied locally.
    Comparisons with missing values are False, as in ADQL."""
    flux = CUTS["flux_over_error_min"]
    keep = (
        (df.parallax >= CUTS["raw_parallax_min_mas"])
        & (df.parallax_over_error > CUTS["parallax_over_error_min"])
        & (df.ruwe < CUTS["ruwe_max"])
        & (df.visibility_periods_used > CUTS["visibility_periods_used_min"])
        & (df.astrometric_excess_noise < CUTS["astrometric_excess_noise_max"])
        & (df.phot_g_mean_flux_over_error > flux)
        & (df.phot_bp_mean_flux_over_error > flux)
        & (df.phot_rp_mean_flux_over_error > flux)
    )
    return df[keep].drop(columns="parallax_over_error")


def download(url, dest, retries=3):
    """Stream url to dest; returns its MD5."""
    for attempt in range(retries):
        try:
            md5 = hashlib.md5()
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
                        md5.update(block)
            return md5.hexdigest()
        except requests.RequestException as exc:
            wait = 30 * (attempt + 1)
            print(f"  {dest.name}: {exc!r}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{dest.name}: download failed after {retries} attempts")


def filter_file(path, chunksize=200_000):
    """Read a bulk ECSV.gz in chunks and apply the cuts. Returns (filtered, n_rows)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        line = f.readline()
        while line.startswith("#"):  # ECSV header
            line = f.readline()
        reader = pd.read_csv(f, names=line.rstrip("\n").split(","), usecols=READ_COLUMNS,
                             na_values=["null"], keep_default_na=False, chunksize=chunksize)
        parts, n_rows = [], 0
        for chunk in reader:
            n_rows += len(chunk)
            parts.append(apply_cuts(chunk))
    filtered = pd.concat(parts, ignore_index=True)
    return filtered[[c for c in READ_COLUMNS if c != "parallax_over_error"]], n_rows


def ingest_file(name, md5_expected, out, tmp):
    """Download, verify, filter, save, delete. Returns the manifest row."""
    t0 = time.time()
    accessed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw_path = tmp / name
    try:
        md5 = download(CDN + name, raw_path)
        if md5 != md5_expected:
            raise RuntimeError(f"{name}: MD5 {md5} != expected {md5_expected}")
        size = raw_path.stat().st_size
        cut, n_rows = filter_file(raw_path)
    finally:
        raw_path.unlink(missing_ok=True)
    unsealed = drop_sealed(cut)
    clean = apply_zero_point(unsealed) if len(unsealed) else unsealed
    clean.to_parquet(out / name.replace(".csv.gz", ".parquet"), index=False)
    return {"file": name, "accessed_utc": accessed, "bytes": size, "md5_ok": True, "rows": n_rows,
            "after_cuts": len(cut), "sealed_dropped": len(cut) - len(unsealed), "kept": len(clean),
            "seconds": round(time.time() - t0)}


def main():
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True, help="Galactic level-0 block")
    ap.add_argument("--out", type=Path, default=Path("data/bulk"))
    ap.add_argument("--limit", type=int, help="only the first N files still to do")
    ap.add_argument("--dry-run", action="store_true", help="list files and total size, download nothing")
    args = ap.parse_args()

    if args.block in json.loads(SPLITS_PATH.read_text())["blocks"]["sealed"]:
        raise SystemExit(f"block {args.block} is sealed (PREREG section 7); refusing to pull it")
    files = list_files()
    names = files_for_block(files, args.block)
    todo = [n for n in names if not (args.out / n.replace(".csv.gz", ".parquet")).exists()]
    total = sum(files[n]["bytes"] for n in names)
    print(f"block {args.block}: {len(names)} of {len(files)} files, {total / 1e9:.1f} GB; {len(todo)} to do")
    if args.dry_run:
        return
    if args.limit:
        todo = todo[:args.limit]
    tmp = args.out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    for name in todo:
        row = ingest_file(name, files[name]["md5"], args.out, tmp)
        pd.DataFrame([row]).to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
        print(f"{name}: {row['bytes'] / 1e6:.0f} MB, rows {row['rows']} -> cuts {row['after_cuts']} "
              f"-> sealed dropped {row['sealed_dropped']} -> kept {row['kept']} ({row['seconds']}s)")


if __name__ == "__main__":
    main()
