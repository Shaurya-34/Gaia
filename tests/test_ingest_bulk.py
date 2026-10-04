"""Offline tests of ingest_bulk.py on synthetic files in the bulk format."""
import gzip
import hashlib

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy_healpix import lonlat_to_healpix

import ingest_bulk as ib
import splits

GOOD = {"source_id": 1, "ra": 10.0, "dec": 5.0, "l": 100.0, "b": 30.0, "parallax": 5.0,
        "parallax_error": 0.05, "parallax_over_error": 100.0, "pmra": 1.0, "pmra_error": 0.05,
        "pmdec": -2.0, "pmdec_error": 0.05, "ruwe": 1.0, "phot_g_mean_mag": 15.0, "bp_rp": 1.0,
        "radial_velocity": None, "radial_velocity_error": None, "visibility_periods_used": 20,
        "astrometric_excess_noise": 0.1, "phot_g_mean_flux_over_error": 500.0,
        "phot_bp_mean_flux_over_error": 100.0, "phot_rp_mean_flux_over_error": 100.0,
        "nu_eff_used_in_astrometry": 1.5, "pseudocolour": None, "ecl_lat": 20.0,
        "astrometric_params_solved": 31, "designation": "Gaia DR3 1", "extra_col": 7}
BAD = [("parallax", 1.8), ("parallax_over_error", 10.0), ("ruwe", 1.4), ("visibility_periods_used", 6),
       ("astrometric_excess_noise", 1.0), ("phot_g_mean_flux_over_error", 10.0),
       ("phot_bp_mean_flux_over_error", None), ("phot_rp_mean_flux_over_error", 10.0)]
TAP_COLUMNS = [c for c in ib.READ_COLUMNS if c not in
               ("parallax_over_error", "nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat",
                "astrometric_params_solved")] + ["parallax_corrected"]


def write_bulk_file(path, rows):
    """Write rows in the bulk format: '#' ECSV header lines, CSV header, 'null' for missing."""
    cols = list(GOOD)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("# %ECSV 1.0\n# ---\n# delimiter: ','\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("null" if r[c] is None else str(r[c]) for c in cols) + "\n")


def rows_good_and_bad():
    rows = [dict(GOOD, source_id=1)]
    for i, (col, val) in enumerate(BAD, start=2):
        rows.append(dict(GOOD, source_id=i, **{col: val}))
    return rows


def test_file_range():
    assert ib.file_range("GaiaSource_003112-005263.csv.gz") == (3112, 5263)


def test_cuts_match_the_archive_query():
    df = pd.DataFrame(rows_good_and_bad()).astype({c: float for c, _ in BAD})
    kept = ib.apply_cuts(df)
    assert kept.source_id.tolist() == [1]
    assert "parallax_over_error" not in kept


def test_filter_file_reads_bulk_format(tmp_path):
    path = tmp_path / "GaiaSource_000000-000001.csv.gz"
    write_bulk_file(path, rows_good_and_bad())
    df, n_rows = ib.filter_file(path, chunksize=3)  # several chunks
    assert n_rows == 1 + len(BAD)
    assert df.source_id.tolist() == [1]
    assert np.isnan(df.radial_velocity.iloc[0])
    assert "designation" not in df and "extra_col" not in df


def test_files_for_block_selects_by_level8_range():
    def level8(l, b):
        c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
        return int(lonlat_to_healpix(c.ra, c.dec, 2 ** 8, order="nested"))

    lon, lat = splits.healpix_to_lonlat(6, 1, order="nested")  # centre of block 6
    inside = level8(lon.deg, lat.deg)
    outside = level8((lon.deg + 180) % 360, -lat.deg)
    names = [f"GaiaSource_{inside:06d}-{inside:06d}.csv.gz", f"GaiaSource_{outside:06d}-{outside:06d}.csv.gz"]
    assert ib.files_for_block(names, 6) == [names[0]]


def fake_download(rows, md5_override=None):
    def _download(url, dest, retries=3):
        write_bulk_file(dest, rows)
        return md5_override or hashlib.md5(dest.read_bytes()).hexdigest()
    return _download


def test_ingest_file_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "download", fake_download(rows_good_and_bad(), md5_override="a" * 32))
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    name = "GaiaSource_000000-000001.csv.gz"
    row = ib.ingest_file(name, "a" * 32, tmp_path, tmp)
    out = pd.read_parquet(tmp_path / "GaiaSource_000000-000001.parquet")
    assert list(out.columns) == TAP_COLUMNS
    assert row["rows"] == 1 + len(BAD) and row["after_cuts"] == 1 and row["kept"] == len(out)
    assert not (tmp / name).exists()  # raw file deleted


def test_md5_mismatch_raises_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "download", fake_download(rows_good_and_bad(), md5_override="0" * 32))
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    name = "GaiaSource_000000-000001.csv.gz"
    with pytest.raises(RuntimeError, match="MD5"):
        ib.ingest_file(name, "f" * 32, tmp_path, tmp)
    assert not (tmp / name).exists()
    assert not (tmp_path / "GaiaSource_000000-000001.parquet").exists()
