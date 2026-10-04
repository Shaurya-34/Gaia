"""Hunt & Reffert 2023 cluster catalogue (VizieR J/A+A/673/A114, tables
`clusters` and `members`; names checked against VizieR on 2026-10-04).

Used for the known-cluster overlap rule (PREREG section 5) and the rediscovery
gate (section 8). Catalogues are test-only: never used to train the vetter.

Run:  python catalogue.py      -> data/catalogues/hr23_clusters.parquet, hr23_members.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from splits import galactic_block

TABLE = "J/A+A/673/A114"
MEMBER_MIN_PLX = 1.5  # mas; members are pre-filtered below our 2 mas cut
OUT = Path(__file__).with_name("data") / "catalogues"
SPLITS_PATH = Path(__file__).with_name("splits.json")
MAX_DIST16_PC = load_config()["data"]["distance_range_pc"][1]  # any part of the cluster inside 500 pc
MEMBER_COLUMNS = ["ID", "Name", "GaiaDR3", "Prob", "inrt", "GLON", "GLAT", "Plx", "pmRA", "pmDE",
                  "Gmag", "BP-RP"]


def fetch(out=OUT):
    """Download the clusters with dist16 < 500 pc and all their members.
    Members are pre-filtered at Plx > MEMBER_MIN_PLX (below our 2 mas cut, so
    no member we could see is lost), then matched to the clusters by ID."""
    from astroquery.vizier import Vizier

    clusters = Vizier(columns=["**"], row_limit=-1, column_filters={"dist16": f"<{MAX_DIST16_PC}"}) \
        .get_catalogs(f"{TABLE}/clusters")[0].to_pandas()
    members = Vizier(columns=MEMBER_COLUMNS, row_limit=-1, column_filters={"Plx": f">{MEMBER_MIN_PLX}"}) \
        .get_catalogs(f"{TABLE}/members")[0].to_pandas()
    members = members[members.ID.isin(clusters.ID)]
    out.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(out / "hr23_clusters.parquet", index=False)
    members.to_parquet(out / "hr23_members.parquet", index=False)
    return clusters, members


def load(out=OUT):
    return pd.read_parquet(out / "hr23_clusters.parquet"), pd.read_parquet(out / "hr23_members.parquet")


def known_overlap(group_ids, injected_ids, members):
    """PREREG section 5: a trial is discarded when its retrieved group holds
    more stars of one known catalogue cluster than of the injected cluster.
    Returns (discard, cluster name or None, stars of that cluster in the group)."""
    in_group = members[members.GaiaDR3.isin(np.asarray(group_ids))]
    if in_group.empty:
        return False, None, 0
    counts = in_group.Name.value_counts()
    n_injected = int(np.isin(np.asarray(group_ids), np.asarray(injected_ids)).sum())
    return bool(counts.iloc[0] > n_injected), counts.index[0], int(counts.iloc[0])


def rediscovery_targets(clusters, members, our_source_ids, min_members=load_config()["rediscovery"]["min_members"]):
    """PREREG section 8 target list: dist50 < 500 pc, at least `min_members`
    members that survive our cuts (present in our data), centre in a train,
    validation or test block (never the sealed one)."""
    sealed = json.loads(SPLITS_PATH.read_text())["blocks"]["sealed"]
    ours = members[members.GaiaDR3.isin(np.asarray(our_source_ids))]
    surviving = ours.groupby("Name").size().rename("n_surviving")
    t = clusters[clusters.dist50 < MAX_DIST16_PC].join(surviving, on="Name")
    t["n_surviving"] = t.n_surviving.fillna(0).astype(int)
    t["block"] = galactic_block(t.GLON.to_numpy(), t.GLAT.to_numpy())
    return t[(t.n_surviving >= min_members) & ~t.block.isin(sealed)]


if __name__ == "__main__":
    c, m = fetch()
    print(f"{len(c)} clusters with dist16 < {MAX_DIST16_PC} pc, {len(m):,} members")
