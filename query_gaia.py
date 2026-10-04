"""Pull nearby Gaia DR3 stars, chunked by HEALPix pixel, with quality cuts and
the Lindegren et al. (2021) parallax zero-point correction.

Run:  python query_gaia.py --level 3 --out data/raw
Resume-safe: pixels whose parquet file already exists are skipped.

NOT YET RUN against the real archive (the dev sandbox could not reach it).
First run on one pixel (--pixels 0) and check the row count and columns.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia

from config import load_config

# Columns we need. nu_eff_used_in_astrometry, pseudocolour, ecl_lat and
# astrometric_params_solved are required by the zero-point package.
COLUMNS = """source_id, ra, dec, l, b, parallax, parallax_error,
    pmra, pmra_error, pmdec, pmdec_error, ruwe,
    phot_g_mean_mag, bp_rp, radial_velocity, radial_velocity_error,
    visibility_periods_used, astrometric_excess_noise,
    phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error,
    nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved"""

# Cuts (PREREG section 2, values in config.yaml). Parallax is cut on the RAW
# value so the archive can prune early; the corrected parallax is cut again locally.
CUTS = load_config()["data"]
MIN_PLX_MAS = CUTS["raw_parallax_min_mas"]
MIN_PLX_OVER_ERROR = CUTS["parallax_over_error_min"]
MAX_RUWE = CUTS["ruwe_max"]
SPLITS_PATH = Path(__file__).with_name("splits.json")
# Gaia archive login (2 lines: username, password). Kept outside the repo.
CREDENTIALS = Path(os.environ.get("GAIA_CREDENTIALS", Path.home() / ".gaia_credentials"))


def pixel_source_id_range(level: int, pixel: int):
    """Gaia source_id encodes the nested HEALPix index at level 12 in its top
    bits, so HEALPix level n = source_id / 2**(59 - 2n)  (Gaia DR2 docs).
    A range filter on source_id is indexed, so it is much faster than calling
    a function in the WHERE clause.
    """
    width = 2 ** (59 - 2 * level)
    return pixel * width, (pixel + 1) * width


def build_query(level: int, pixel: int) -> str:
    lo, hi = pixel_source_id_range(level, pixel)
    return f"""
    SELECT {COLUMNS}
    FROM gaiadr3.gaia_source
    WHERE source_id >= {lo} AND source_id < {hi}
      AND parallax >= {MIN_PLX_MAS}
      AND parallax_over_error > {MIN_PLX_OVER_ERROR}
      AND ruwe < {MAX_RUWE}
      AND visibility_periods_used > {CUTS["visibility_periods_used_min"]}
      AND astrometric_excess_noise < {CUTS["astrometric_excess_noise_max"]}
      AND phot_g_mean_flux_over_error > {CUTS["flux_over_error_min"]}
      AND phot_bp_mean_flux_over_error > {CUTS["flux_over_error_min"]}
      AND phot_rp_mean_flux_over_error > {CUTS["flux_over_error_min"]}
    """


def apply_zero_point(df: pd.DataFrame) -> pd.DataFrame:
    """Add parallax_corrected = parallax - zpt. The zero-point model is only
    calibrated for 6 < G < 21 and for 5- and 6-parameter solutions; rows
    outside that are dropped rather than silently left uncorrected.
    """
    from zero_point import zpt

    zpt.load_tables()
    ok = (
        df.astrometric_params_solved.isin(CUTS["astrometric_params_solved"])
        & df.phot_g_mean_mag.between(*CUTS["g_mag_range"])
    )
    df = df[ok].copy()
    zp = zpt.get_zpt(
        df.phot_g_mean_mag.to_numpy(),
        df.nu_eff_used_in_astrometry.to_numpy(),
        df.pseudocolour.to_numpy(),
        df.ecl_lat.to_numpy(),
        df.astrometric_params_solved.to_numpy(),
        _warnings=False,
    )
    df["parallax_corrected"] = df.parallax - zp
    df = df[df.parallax_corrected >= CUTS["parallax_corrected_min_mas"]]  # ~500 pc
    return df.drop(columns=["nu_eff_used_in_astrometry", "pseudocolour",
                            "ecl_lat", "astrometric_params_solved"])


def drop_sealed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stars in the sealed block (Galactic level-0, from splits.json)
    before anything is written. Download chunks are ICRS, so this is per star."""
    from splits import galactic_block

    sealed = json.loads(SPLITS_PATH.read_text())["blocks"]["sealed"]
    return df[~np.isin(galactic_block(df.l.to_numpy(), df.b.to_numpy()), sealed)]


def fetch_pixel(level: int, pixel: int, retries: int = 3) -> pd.DataFrame:
    query = build_query(level, pixel)
    for attempt in range(retries):
        try:
            job = Gaia.launch_job_async(query, dump_to_file=False)
            return job.get_results().to_pandas()
        except Exception as exc:  # network / archive hiccups are common
            wait = 30 * (attempt + 1)
            print(f"  pixel {pixel}: {exc!r}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"pixel {pixel} failed after {retries} attempts")


def icrs_pixels_for_block(block: int, level: int, sub: int = 3) -> list:
    """ICRS nested pixels at `level` (the download chunks) that overlap the
    Galactic level-0 `block`. Each pixel is tested at its 4**sub children; the
    neighbours of every hit are added too, so thin slivers of the block that
    no sample point lands in are not missed."""
    from astropy.coordinates import SkyCoord
    from astropy_healpix import healpix_to_lonlat, neighbours
    from splits import galactic_block

    child = np.arange(12 * 4 ** (level + sub))
    ra, dec = healpix_to_lonlat(child, 2 ** (level + sub), order="nested")
    g = SkyCoord(ra=ra, dec=dec, frame="icrs").galactic
    hits = np.unique(child[galactic_block(g.l.deg, g.b.deg) == block] // 4 ** sub)
    nb = neighbours(hits, 2 ** level, order="nested")
    return sorted(int(p) for p in np.unique(np.r_[hits, nb[nb >= 0]]))


def login():
    if CREDENTIALS.exists():
        Gaia.login(credentials_file=str(CREDENTIALS))  # longer timeouts, no row cap
    else:
        print(f"no {CREDENTIALS}; querying anonymously")


def pull_pixel(level: int, pix: int, out: Path) -> dict:
    """Fetch, clean and save one chunk; returns its manifest row."""
    t0 = time.time()
    accessed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw = fetch_pixel(level, pix)
    unsealed = drop_sealed(raw)
    clean = apply_zero_point(unsealed) if len(unsealed) else unsealed
    clean.to_parquet(out / f"L{level}_{pix:05d}.parquet", index=False)
    return {"level": level, "pixel": pix, "accessed_utc": accessed, "raw": len(raw),
            "sealed_dropped": len(raw) - len(unsealed), "kept": len(clean),
            "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3, help="HEALPix level (3 -> 768 pixels)")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--pixels", type=int, nargs="*", help="subset of pixels (for testing)")
    ap.add_argument("--block", type=int, help="all chunks overlapping this Galactic level-0 block")
    ap.add_argument("--workers", type=int, default=1, help="parallel archive jobs (one process each)")
    args = ap.parse_args()

    if args.block is not None and args.block in json.loads(SPLITS_PATH.read_text())["blocks"]["sealed"]:
        raise SystemExit(f"block {args.block} is sealed (PREREG section 7); refusing to pull it")
    args.out.mkdir(parents=True, exist_ok=True)
    n_pix = 12 * 4 ** args.level
    if args.pixels:
        pixels = args.pixels
    elif args.block is not None:
        pixels = icrs_pixels_for_block(args.block, args.level)
    else:
        pixels = range(n_pix)
    todo = [p for p in pixels if not (args.out / f"L{args.level}_{p:05d}.parquet").exists()]
    print(f"{len(todo)} of {len(pixels)} chunks to pull at level {args.level}, {args.workers} worker(s)")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    manifest = args.out / "manifest.csv"
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=login) as pool:
        futures = {pool.submit(pull_pixel, args.level, p, args.out): p for p in todo}
        for fut in as_completed(futures):
            pix = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                failed.append(pix)
                print(f"pixel {pix}: FAILED {exc!r}")
                continue
            pd.DataFrame([row]).to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
            print(f"pixel {pix}/{n_pix}: raw {row['raw']} -> sealed dropped {row['sealed_dropped']} "
                  f"-> kept {row['kept']} ({row['seconds']}s)")
    if failed:
        raise SystemExit(f"{len(failed)} chunk(s) failed (rerun to retry): {sorted(failed)}")


if __name__ == "__main__":
    main()
