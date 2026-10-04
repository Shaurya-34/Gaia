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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3, help="HEALPix level (3 -> 768 pixels)")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--pixels", type=int, nargs="*", help="subset of pixels (for testing)")
    args = ap.parse_args()

    if CREDENTIALS.exists():
        Gaia.login(credentials_file=str(CREDENTIALS))  # longer timeouts, no row cap
    else:
        print(f"no {CREDENTIALS}; querying anonymously")
    args.out.mkdir(parents=True, exist_ok=True)
    n_pix = 12 * 4 ** args.level
    pixels = args.pixels if args.pixels else range(n_pix)

    manifest = args.out / "manifest.csv"
    for pix in pixels:
        path = args.out / f"L{args.level}_{pix:05d}.parquet"
        if path.exists():
            continue
        t0 = time.time()
        accessed = datetime.now(timezone.utc).isoformat(timespec="seconds")
        raw = fetch_pixel(args.level, pix)
        unsealed = drop_sealed(raw)
        clean = apply_zero_point(unsealed) if len(unsealed) else unsealed
        clean.to_parquet(path, index=False)
        row = pd.DataFrame([{"level": args.level, "pixel": pix, "accessed_utc": accessed,
                             "raw": len(raw), "sealed_dropped": len(raw) - len(unsealed),
                             "kept": len(clean)}])
        row.to_csv(manifest, mode="a", header=not manifest.exists(), index=False)
        print(f"pixel {pix}/{n_pix}: raw {len(raw)} -> sealed dropped "
              f"{len(raw) - len(unsealed)} -> kept {len(clean)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
