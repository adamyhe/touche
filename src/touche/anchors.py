from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_bed_anchors(path: str | Path) -> pd.DataFrame:
    """Read BED3/BED4 anchors and add an integer center column."""

    data = pd.read_csv(path, sep="\t", header=None, comment="#")
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three BED columns in {path}")
    anchors = data.iloc[:, :4].copy() if data.shape[1] >= 4 else data.iloc[:, :3].copy()
    if anchors.shape[1] == 3:
        anchors[3] = "."
    anchors.columns = ["chr", "start", "end", "strand"]
    anchors["start"] = anchors["start"].astype(int)
    anchors["end"] = anchors["end"].astype(int)
    anchors["center"] = ((anchors["start"] + anchors["end"]) // 2).astype(int)
    return anchors
