"""Loading open datasets and building simulation "populations".

A population is the metric values of the dataset's control group plus covariates.
Simulated samples are drawn from it with replacement, i.e. the empirical
control distribution is treated as the population (an assumption).
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

HILLSTROM_URL = ("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_"
                 "E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
HILLSTROM_FILE = DATA_DIR / "hillstrom.csv"
COOKIE_FILE = DATA_DIR / "cookie_cats.csv"

CONTROL_LABEL = {"hillstrom": "No E-Mail", "synthetic": "No E-Mail", "cookie_cats": "gate_30"}

METRIC_KIND = {
    "history": "continuous", "spend": "zero_inflated", "visit": "binary",
    "conversion": "binary", "sum_gamerounds": "continuous",
    "retention_1": "binary", "retention_7": "binary",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_hillstrom(force: bool = False, timeout: int = 600) -> Path:
    """Download the CSV from MineThatData (about 4 MB; the server can be slow)."""
    DATA_DIR.mkdir(exist_ok=True)
    if HILLSTROM_FILE.exists() and not force:
        return HILLSTROM_FILE
    tmp = HILLSTROM_FILE.with_suffix(".part")
    with urllib.request.urlopen(HILLSTROM_URL, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.rename(HILLSTROM_FILE)
    return HILLSTROM_FILE


def load_hillstrom() -> pd.DataFrame:
    if not HILLSTROM_FILE.exists():
        download_hillstrom()
    df = pd.read_csv(HILLSTROM_FILE)
    need = {"recency", "history", "mens", "womens", "zip_code", "newbie",
            "channel", "segment", "visit", "conversion", "spend"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Hillstrom file is missing columns: {missing}")
    df["group"] = df["segment"]
    return df


def load_cookie_cats() -> pd.DataFrame:
    """The Kaggle Cookie Cats dataset must be downloaded manually to data/cookie_cats.csv."""
    if not COOKIE_FILE.exists():
        raise FileNotFoundError(
            "Put cookie_cats.csv into data/ (Kaggle: 'Mobile Games A/B Testing - "
            "Cookie Cats'). Without it the experiments run on Hillstrom.")
    df = pd.read_csv(COOKIE_FILE)
    df["group"] = df["version"]
    return df


def make_synthetic(n: int = 60000, seed: int = 7) -> pd.DataFrame:
    """Synthetic data with known structure: metrics depend on covariates,
    so MAR missingness really shifts the means. Parameters are hand-picked
    and documented in the paper as assumptions."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 13, n)
    channel = rng.choice(["Phone", "Web", "Multichannel"], n, p=[0.44, 0.44, 0.12])
    newbie = rng.integers(0, 2, n)
    lp = 4.6 - 0.06 * (recency - 6) + 0.4 * (channel == "Multichannel") - 0.3 * newbie
    history = np.exp(lp + rng.normal(0.0, 0.9, n))
    p_visit = 1 / (1 + np.exp(-(-1.9 - 0.09 * (recency - 6) + 0.3 * (channel == "Web"))))
    visit = (rng.random(n) < p_visit).astype(int)
    buy = (visit == 1) & (rng.random(n) < 0.06)
    spend = np.where(buy, np.exp(rng.normal(4.6, 0.7, n)), 0.0)
    return pd.DataFrame({
        "recency": recency, "channel": channel, "newbie": newbie,
        "history": history, "visit": visit, "conversion": buy.astype(int),
        "spend": spend, "group": "No E-Mail",
    })


def load_source(source: str) -> pd.DataFrame:
    if source == "hillstrom":
        return load_hillstrom()
    if source == "synthetic":
        return make_synthetic()
    if source == "cookie_cats":
        return load_cookie_cats()
    raise ValueError(source)


@dataclass
class Pool:
    source: str
    metric: str
    kind: str
    y: np.ndarray
    strata: np.ndarray
    n_strata: int
    mar_lp: np.ndarray | None
    mu: float = 0.0
    sd: float = 0.0
    q999: float = 0.0
    y_sorted: np.ndarray = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.mu = float(self.y.mean())
        self.sd = float(self.y.std(ddof=1))
        self.q999 = float(np.quantile(self.y, 0.999))
        self.y_sorted = np.sort(self.y)

    @property
    def size(self) -> int:
        return len(self.y)


def covariates(df: pd.DataFrame):
    """Strata for post-stratification and the linear predictor for MAR missingness."""
    if not {"recency", "channel", "newbie"} <= set(df.columns):
        return np.zeros(len(df), dtype=np.int64), 1, None
    rec = df["recency"].to_numpy(dtype=float)
    rec_bin = np.digitize(rec, [4.5, 8.5])                       # 0, 1, 2
    ch = pd.Categorical(df["channel"], categories=["Phone", "Web", "Multichannel"]).codes
    nb = df["newbie"].to_numpy(dtype=np.int64)
    strata = (ch * 6 + nb * 3 + rec_bin).astype(np.int64)        # 18 strata
    c = config.MAR_COEF
    rec_z = (rec - rec.mean()) / rec.std()
    lp = c["recency"] * rec_z + c["phone"] * (df["channel"].to_numpy() == "Phone") + c["newbie"] * nb
    return strata, 18, lp.astype(float)


def build_pool(source: str, metric: str) -> Pool:
    df = load_source(source)
    ctrl = df[df["group"] == CONTROL_LABEL[source]].reset_index(drop=True)
    strata, k, lp = covariates(ctrl)
    return Pool(source=source, metric=metric, kind=METRIC_KIND[metric],
                y=ctrl[metric].to_numpy(dtype=float), strata=strata, n_strata=k, mar_lp=lp)


_POOLS: dict = {}


def get_pool(source: str, metric: str) -> Pool:
    key = (source, metric)
    if key not in _POOLS:
        _POOLS[key] = build_pool(source, metric)
    return _POOLS[key]
