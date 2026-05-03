"""Modal app for mars-ice data ingestion and processing.

All heavy work runs here. Local invocation: `uv run modal run modal_app.py::hello`.
"""

from __future__ import annotations

import hashlib
import pathlib
import time
from dataclasses import dataclass

import modal

app = modal.App("mars-ice")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "rasterio>=1.5",
    "pyproj>=3.7",
    "numpy>=2.0",
    "polars>=1.0",
    "httpx>=0.27",
    "pdr>=1.4",
    "lightgbm>=4.5",
    "scikit-learn>=1.5",
    "openpyxl>=3.1",
    "pdfplumber>=0.11",
    "fastexcel>=0.12",
)

volume = modal.Volume.from_name("mars-ice-data", create_if_missing=True)
DATA_DIR = "/data"
RAW = f"{DATA_DIR}/raw"


@app.function(image=image, volumes={DATA_DIR: volume})
def hello() -> dict:
    """Sanity check: write a file to the volume and report back."""
    import os
    import sys

    root = pathlib.Path(DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "hello.txt"
    marker.write_text(f"hello from mars-ice modal app at {time.time()}\n")
    volume.commit()

    contents = sorted(p.name for p in root.iterdir())
    return {
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "data_dir_contents": contents,
        "marker_size": marker.stat().st_size,
    }


@dataclass
class LayerSpec:
    name: str
    url: str
    subdir: str  # under raw/
    filename_override: str | None = None  # set when URL has query strings or non-trivial paths

    @property
    def filename(self) -> str:
        if self.filename_override:
            return self.filename_override
        # strip query string then take basename
        base = self.url.split("?")[0].rsplit("/", 1)[-1]
        return base


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=3600)
def download_one(spec_dict: dict) -> dict:
    """Download a single product into the volume. Idempotent. Retries on transient errors."""
    import httpx

    spec = LayerSpec(**spec_dict)
    dest_dir = pathlib.Path(RAW) / spec.subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.filename
    tmp = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        return {
            "name": spec.name,
            "url": spec.url,
            "path": str(dest),
            "size": dest.stat().st_size,
            "skipped": True,
            "ok": True,
        }

    last_err = None
    for attempt in range(1, 4):
        sha = hashlib.sha256()
        n = 0
        expected: int | None = None
        t0 = time.time()
        try:
            timeout = httpx.Timeout(connect=30, read=120, write=60, pool=30)
            with httpx.stream("GET", spec.url, timeout=timeout, follow_redirects=True) as r:
                r.raise_for_status()
                cl = r.headers.get("content-length")
                expected = int(cl) if cl and cl.isdigit() else None
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1024 * 1024):
                        f.write(chunk)
                        sha.update(chunk)
                        n += len(chunk)
            if expected is not None and n != expected:
                raise httpx.RemoteProtocolError(f"truncated: got {n} bytes, expected {expected}")
            tmp.replace(dest)
            elapsed = time.time() - t0
            volume.commit()
            return {
                "name": spec.name,
                "url": spec.url,
                "path": str(dest),
                "size": n,
                "sha256": sha.hexdigest()[:16],
                "elapsed_s": round(elapsed, 1),
                "mb_per_s": round(n / 1024 / 1024 / max(elapsed, 1e-3), 2),
                "skipped": False,
                "attempts": attempt,
                "ok": True,
            }
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            last_err = f"{type(e).__name__}: {e}"
            tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)

    return {
        "name": spec.name,
        "url": spec.url,
        "ok": False,
        "error": last_err,
        "attempts": 3,
    }


@app.function(image=image, volumes={DATA_DIR: volume})
def list_volume() -> list[dict]:
    """Walk the volume and report sizes."""
    out = []
    root = pathlib.Path(DATA_DIR)
    for p in root.rglob("*"):
        if p.is_file():
            out.append({"path": str(p.relative_to(root)), "size_mb": round(p.stat().st_size / 1024 / 1024, 3)})
    return sorted(out, key=lambda d: d["path"])


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=600)
def inspect_layers() -> list[dict]:
    """Open every raster/table on the volume and report what we can about it."""
    import polars as pl
    import rasterio
    from rasterio.errors import RasterioIOError

    root = pathlib.Path(RAW)
    results: list[dict] = []

    # Rasters: SWIM GeoTIFFs and MOLA IMG (via .lbl pairing)
    for tif in sorted(root.rglob("*.tif")):
        try:
            with rasterio.open(tif) as ds:
                results.append({
                    "kind": "geotiff",
                    "path": str(tif.relative_to(root)),
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtypes[0]),
                    "crs": str(ds.crs),
                    "auth": ds.crs.to_authority() if ds.crs else None,
                    "bounds": [round(b, 1) for b in ds.bounds],
                    "res": [round(r, 1) for r in ds.res],
                    "nodata": ds.nodata,
                })
        except RasterioIOError as e:
            results.append({"kind": "geotiff", "path": str(tif.relative_to(root)), "error": str(e)})

    # MOLA MEGDR via .lbl (PDS3) — rasterio's PDS driver
    for lbl in sorted(root.rglob("*.lbl")):
        try:
            with rasterio.open(lbl) as ds:
                results.append({
                    "kind": "pds_img",
                    "path": str(lbl.relative_to(root)),
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtypes[0]),
                    "crs": str(ds.crs),
                    "bounds": [round(b, 1) for b in ds.bounds],
                    "res": [round(r, 4) for r in ds.res],
                    "nodata": ds.nodata,
                })
        except RasterioIOError as e:
            results.append({"kind": "pds_img", "path": str(lbl.relative_to(root)), "error": str(e)[:200]})

    # Neutron / GRS .tab tables
    for tab in sorted(root.rglob("*.tab")):
        try:
            df = pl.read_csv(
                tab,
                separator=",",
                has_header=False,
                new_columns=["lon", "lat", "value"],
            ).with_columns(pl.col(c).str.strip_chars().cast(pl.Float64) for c in ["lon", "lat", "value"])
            v = df["value"]
            results.append({
                "kind": "tab",
                "path": str(tab.relative_to(root)),
                "rows": df.height,
                "lon": [round(df["lon"].min(), 2), round(df["lon"].max(), 2)],
                "lat": [round(df["lat"].min(), 2), round(df["lat"].max(), 2)],
                "value": [round(v.min(), 3), round(v.mean(), 3), round(v.max(), 3)],
            })
        except Exception as e:
            results.append({"kind": "tab", "path": str(tab.relative_to(root)), "error": str(e)[:200]})

    return results


@app.local_entrypoint()
def inspect_main():
    for r in inspect_layers.remote():
        print(r)


# --- entry points for local invocation -----------------------------------------------------------


@app.local_entrypoint()
def hello_main():
    """`uv run modal run modal_app.py::hello_main`"""
    result = hello.remote()
    print(result)


@app.local_entrypoint()
def smoke_download():
    """Pull one small known-good product to verify the download function end-to-end."""
    spec = LayerSpec(
        name="swim4mim_ci_0_1",
        url="https://swim.psi.edu/output/SWIM4MIM/Global/Composite/SWIM4MIM_Ci_0_1.tif",
        subdir="swim4mim",
    )
    result = download_one.remote(spec.__dict__)
    print(result)


# Canonical layer manifest. MB-scale items, public archives, no auth required.
SWIM_BASE = "https://swim.psi.edu/output"
PDS = "https://pds-geosciences.wustl.edu"
MOLA_BASE = f"{PDS}/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x"
GRS_BASE = f"{PDS}/ody/urn-nasa-pds-odyssey_grs_special/data"

MANIFEST: list[LayerSpec] = [
    # Topography — MOLA MEGDR 16 px/° global single tile (~33 MB)
    LayerSpec("mola_megt_16", f"{MOLA_BASE}/meg016/megt90n000eb.img", "mola_megdr"),
    LayerSpec("mola_megt_16_lbl", f"{MOLA_BASE}/meg016/megt90n000eb.lbl", "mola_megdr"),
    # Areoid (geoid reference)
    LayerSpec("mola_mega_16", f"{MOLA_BASE}/meg016/mega90n000eb.img", "mola_megdr"),
    LayerSpec("mola_mega_16_lbl", f"{MOLA_BASE}/meg016/mega90n000eb.lbl", "mola_megdr"),
    # SWIM4MIM (mid-lat, 2024+) — three depth bins of composite ice consistency
    LayerSpec("swim4mim_ci_0_1", f"{SWIM_BASE}/SWIM4MIM/Global/Composite/SWIM4MIM_Ci_0_1.tif", "swim4mim"),
    LayerSpec("swim4mim_ci_1_5", f"{SWIM_BASE}/SWIM4MIM/Global/Composite/SWIM4MIM_Ci_1_5.tif", "swim4mim"),
    LayerSpec("swim4mim_ci_5", f"{SWIM_BASE}/SWIM4MIM/Global/Composite/SWIM4MIM_Ci_5.tif", "swim4mim"),
    # SWIM 2.0 (2020) — extends polar coverage SWIM4MIM lacks
    LayerSpec("swim2_c0_1", f"{SWIM_BASE}/SWIM2/Global/Composite/SWIM2_c0_1.tif", "swim2"),
    LayerSpec("swim2_c1_5", f"{SWIM_BASE}/SWIM2/Global/Composite/SWIM2_c1_5.tif", "swim2"),
    LayerSpec("swim2_c_5", f"{SWIM_BASE}/SWIM2/Global/Composite/SWIM2_c_5.tif", "swim2"),
    # Odyssey neutron flux (Feldman 2002 Science paper supporting data) — placeholder for WEH
    LayerSpec("ns_epithermal", f"{GRS_BASE}/ns_epithermal_020917.tab", "odyssey_grs"),
    LayerSpec("ns_epithermal_xml", f"{GRS_BASE}/ns_epithermal_020917.xml", "odyssey_grs"),
    LayerSpec("ns_thermal", f"{GRS_BASE}/ns_thermal_020917.tab", "odyssey_grs"),
    LayerSpec("ns_thermal_xml", f"{GRS_BASE}/ns_thermal_020917.xml", "odyssey_grs"),
    LayerSpec("ns_fast", f"{GRS_BASE}/ns_fast_020917.tab", "odyssey_grs"),
    LayerSpec("ns_fast_xml", f"{GRS_BASE}/ns_fast_020917.xml", "odyssey_grs"),
    # --- Label catalogs ---
    # Dundas et al. 2021 USGS ScienceBase data release (DOI 10.5066/P9Y8FR1R) — shallow exposures.
    # ScienceBase uses opaque content-hash query params, not filenames. URLs from the JSON API.
    LayerSpec(
        "dundas_2021_TableS1",
        "https://www.sciencebase.gov/catalog/file/get/5f99f0b8d34e198cb786e8e7?f=__disk__32%2F02%2F21%2F3202211d40bdb979905fb1ba8c5a93a3881287e9",
        "labels/dundas_2021",
        filename_override="TableS1_final.csv",
    ),
    LayerSpec(
        "dundas_2021_TableS2",
        "https://www.sciencebase.gov/catalog/file/get/5f99f0b8d34e198cb786e8e7?f=__disk__ba%2F28%2F98%2Fba28985a644727d7c088248d81a3b43b1533ba49",
        "labels/dundas_2021",
        filename_override="TableS2_final.csv",
    ),
    LayerSpec(
        "dundas_2021_TableS3",
        "https://www.sciencebase.gov/catalog/file/get/5f99f0b8d34e198cb786e8e7?f=__disk__30%2F72%2Fe1%2F3072e1d28ff6e2ffbb0c08667f9509f28e8b0d59",
        "labels/dundas_2021",
        filename_override="TableS3_final.csv",
    ),
    LayerSpec(
        "dundas_2021_TableS4",
        "https://www.sciencebase.gov/catalog/file/get/5f99f0b8d34e198cb786e8e7?f=__disk__cc%2F0f%2Feb%2Fcc0febbe29e84fe44db8017e4c737a9b958f6b0f",
        "labels/dundas_2021",
        filename_override="TableS4_final.csv",
    ),
    # McGlasson et al. 2024 NPLD figshare bundle (2.6 GB tar with reflector CSVs inside)
    LayerSpec(
        "mcglasson_2024_bundle",
        "https://figshare.com/ndownloader/files/44729680",
        "labels/mcglasson_2024",
        filename_override="mcglasson_2024_bundle.tar",
    ),
    # --- Deep ice catalogs (>5 m) ---
    # Daubar et al. 2022 — 1,203 dated impacts; ~6% expose subsurface ice with depth-of-excavation.
    # Zenodo deposit (CC-BY 4.0), single Excel file with the master catalog.
    LayerSpec(
        "daubar_2022_table",
        "https://zenodo.org/records/6604912/files/Daubar_2022_catalog_tableS1.xlsx",
        "labels/daubar_2022",
        filename_override="daubar_2022_tableS1.xlsx",
    ),
    # Bramson et al. 2015 GRL — Arcadia Planitia SHARAD + terraced craters. arXiv preprint is
    # auth-free; same content as the paywalled Wiley version.
    LayerSpec(
        "bramson_2015_pdf",
        "https://arxiv.org/pdf/1509.03210",
        "labels/bramson_2015",
        filename_override="bramson_2015.pdf",
    ),
    # Stuurman 2016 (Utopia Planitia SHARAD) and Petersen 2018 (Deuteronilus LDAs) deliberately
    # NOT included — Wiley blocks programmatic PDF access and no open mirrors exist. We refuse
    # to fabricate them as bounding boxes (would repeat the v1 "model thinks ice = looks like
    # the cluster I trained on" failure mode). If we want them later: manual coordinate
    # transcription from LPSC abstracts, or shipping a curated JSON of pick centroids.
]


@app.local_entrypoint()
def pull_manifest():
    """Parallel-download every layer in MANIFEST onto the Modal volume."""
    specs = [s.__dict__ for s in MANIFEST]
    print(f"submitting {len(specs)} downloads...")
    results = list(download_one.map(specs, return_exceptions=True))
    total = 0
    n_ok = n_skip = n_fail = 0
    for r in results:
        if isinstance(r, BaseException):
            print(f"  FAIL  <unmapped exception>  {r}")
            n_fail += 1
            continue
        if not r.get("ok"):
            print(f"  FAIL  {r['name']:<22}  {r.get('error')}")
            n_fail += 1
            continue
        size = r["size"]
        total += size
        if r.get("skipped"):
            print(f"  SKIP  {r['name']:<22}  {size / 1024 / 1024:>8.2f} MB  (on volume)")
            n_skip += 1
        else:
            print(f"  {r['mb_per_s']:>5} MB/s  {r['name']:<22}  {size / 1024 / 1024:>8.2f} MB  attempts={r['attempts']}")
            n_ok += 1
    print(
        f"\n{len(results)} files: {n_ok} downloaded, {n_skip} skipped, {n_fail} failed; "
        f"{total / 1024 / 1024:.2f} MB total"
    )


@app.local_entrypoint()
def show_volume():
    for entry in list_volume.remote():
        print(f"{entry['size_mb']:>10.3f} MB  {entry['path']}")


@app.function(image=image, volumes={DATA_DIR: volume})
def delete_paths(rel_paths: list[str]) -> list[str]:
    """Delete specific files from the volume by raw/-relative path."""
    deleted = []
    for rel in rel_paths:
        p = pathlib.Path(RAW) / rel
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    if deleted:
        volume.commit()
    return deleted


@app.local_entrypoint()
def remove(*paths: str):
    """`uv run modal run modal_app.py::remove swim4mim/SWIM4MIM_Ci_5.tif swim2/SWIM2_c_5.tif`"""
    print(delete_paths.remote(list(paths)))


@app.function(image=image, volumes={DATA_DIR: volume})
def move_path(src_rel: str, dst_rel: str) -> dict:
    """Move/rename a file under raw/. If dst exists, it's overwritten."""
    src = pathlib.Path(RAW) / src_rel
    dst = pathlib.Path(RAW) / dst_rel
    if not src.exists():
        return {"ok": False, "reason": f"src does not exist: {src}"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    volume.commit()
    return {"ok": True, "src": str(src), "dst": str(dst), "size_mb": round(dst.stat().st_size / 1024 / 1024, 2)}


@app.local_entrypoint()
def mv(src: str, dst: str):
    print(move_path.remote(src, dst))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def inspect_label_sources() -> dict:
    """Read Dundas CSV headers and list McGlasson tar contents (CSVs only)."""
    import tarfile

    import polars as pl

    out: dict = {}

    # Dundas CSVs
    dundas_dir = pathlib.Path(RAW) / "labels/dundas_2021"
    out["dundas_2021"] = {}
    for csv in sorted(dundas_dir.glob("*.csv")):
        try:
            df = pl.read_csv(csv, has_header=True, infer_schema_length=200)
        except Exception as e:
            out["dundas_2021"][csv.name] = {"error": str(e)[:200]}
            continue
        out["dundas_2021"][csv.name] = {
            "rows": df.height,
            "columns": df.columns,
            "head": df.head(3).to_dicts(),
        }

    # McGlasson zip — list CSV entries with sizes, no extract yet (figshare bundle is a zip not tar)
    import zipfile
    zip_path = pathlib.Path(RAW) / "labels/mcglasson_2024/mcglasson_2024_bundle.zip"
    out["mcglasson_2024_zip"] = {"path": str(zip_path), "size_mb": round(zip_path.stat().st_size / 1024 / 1024, 1)}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            csv_members = [m for m in members if m.filename.lower().endswith(".csv")]
            out["mcglasson_2024_zip"]["csv_count"] = len(csv_members)
            out["mcglasson_2024_zip"]["csv_files"] = [
                {"name": m.filename, "size_kb": round(m.file_size / 1024, 1)} for m in csv_members
            ]
            all_top = sorted({m.filename.split("/")[0] for m in members})
            out["mcglasson_2024_zip"]["top_level"] = all_top
            out["mcglasson_2024_zip"]["total_members"] = len(members)
    except Exception as e:
        out["mcglasson_2024_zip"]["error"] = str(e)[:200]

    return out


@app.local_entrypoint()
def inspect_labels_main():
    import json
    print(json.dumps(inspect_label_sources.remote(), indent=2, default=str))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def inspect_deep_label_sources() -> dict:
    """Peek at Daubar 2022 Excel and Bramson 2015 PDF contents."""
    import polars as pl
    import pdfplumber

    out: dict = {}

    # Daubar 2022 — Excel with crater catalog. Convert to dict by reading sheets.
    daubar_path = pathlib.Path(RAW) / "labels/daubar_2022/daubar_2022_tableS1.xlsx"
    out["daubar_2022"] = {"path": str(daubar_path), "size_kb": round(daubar_path.stat().st_size / 1024, 1)}
    try:
        # polars can read xlsx via openpyxl as engine
        df = pl.read_excel(daubar_path)
        ice_col = "Ice-exposing impact"
        out["daubar_2022"]["sheet_default"] = {
            "rows": df.height,
            "columns": df.columns[:30],
            "head": df.head(3).to_dicts(),
            "ice_unique": df[ice_col].value_counts().sort("count", descending=True).to_dicts() if ice_col in df.columns else None,
        }
    except Exception as e:
        out["daubar_2022"]["error"] = str(e)[:300]

    # Bramson 2015 — PDF, look for table-like content. List pages, extract text from candidate pages.
    bramson_path = pathlib.Path(RAW) / "labels/bramson_2015/bramson_2015.pdf"
    out["bramson_2015"] = {"path": str(bramson_path), "size_kb": round(bramson_path.stat().st_size / 1024, 1)}
    try:
        with pdfplumber.open(bramson_path) as pdf:
            out["bramson_2015"]["n_pages"] = len(pdf.pages)
            page_summaries = []
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                text = (page.extract_text() or "")
                head = text[:300].replace("\n", " | ")
                page_summaries.append({
                    "page": i + 1,
                    "has_tables": bool(tables),
                    "table_dims": [(len(t), len(t[0]) if t else 0) for t in tables],
                    "text_head": head,
                    "text_len": len(text),
                })
            out["bramson_2015"]["pages"] = page_summaries
    except Exception as e:
        out["bramson_2015"]["error"] = str(e)[:300]

    return out


@app.local_entrypoint()
def inspect_deep_main():
    import json
    print(json.dumps(inspect_deep_label_sources.remote(), indent=2, default=str))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def file_magic(rel_path: str, n_bytes: int = 512) -> dict:
    """Read first N bytes of a file under raw/ to identify format."""
    p = pathlib.Path(RAW) / rel_path
    if not p.exists():
        return {"ok": False, "reason": f"not found: {p}"}
    with open(p, "rb") as f:
        head = f.read(n_bytes)
    return {
        "path": str(p),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
        "head_hex": head[:64].hex(),
        "head_text": head[:200].decode("ascii", errors="replace"),
    }


@app.local_entrypoint()
def magic(rel_path: str):
    import json
    print(json.dumps(file_magic.remote(rel_path), indent=2))


FEATURE_COLS_SHALLOW = [
    "mola_topography_m",
    "mola_areoid_m",
    "ns_epithermal_cps",
    "ns_thermal_cps",
    "ns_fast_cps",
    "swim4mim_ci_0_1m",
    "swim4mim_ci_1_5m",
    "swim4mim_ci_5m",
    "swim2_c_0_1m",
    "swim2_c_1_5m",
    "swim2_c_5m",
]

# When training on SWIM ice-consistency as the SOFT LABEL TARGET, we must NOT
# use SWIM channels as features — otherwise the model trivially copies the
# label. These are the non-SWIM features used by the SWIM-soft-label model.
FEATURE_COLS_NONSWIM = [
    "mola_topography_m",
    "ns_epithermal_cps",
    "ns_thermal_cps",
    "ns_fast_cps",
]

SWIM_SHALLOW_DEPTH_KEYS = [
    "swim4mim_ci_0_1m",
    "swim4mim_ci_1_5m",
    "swim2_c_0_1m",
    "swim2_c_1_5m",
]

# SWIM 5m+ ice-consistency channels — published expert assessment of deep ice
# down to ~hundreds of meters where SHARAD reflectors and geomorphic evidence
# constrain it. Same merging logic as the shallow target.
SWIM_DEEP_DEPTH_KEYS = [
    "swim4mim_ci_5m",
    "swim2_c_5m",
]


def _spatial_blocks(lat: "np.ndarray", lon: "np.ndarray", n_lat: int = 6, n_lon: int = 6) -> "np.ndarray":
    """Assign each point to a coarse lat/lon block id. Blocks are kept whole during CV."""
    import numpy as np
    lat_bins = np.linspace(-90, 90, n_lat + 1)
    lon_bins = np.linspace(0, 360, n_lon + 1)
    lat_idx = np.clip(np.digitize(lat, lat_bins) - 1, 0, n_lat - 1)
    lon_idx = np.clip(np.digitize(lon % 360, lon_bins) - 1, 0, n_lon - 1)
    return lat_idx * n_lon + lon_idx


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=600)
def train_shallow_baseline(
    n_lat: int = 6,
    n_lon: int = 6,
    min_weight: float = 0.5,
) -> dict:
    """Train a calibrated LightGBM on the shallow labels with spatial-block CV.

    For each fold: train on labels outside one spatial block, predict on labels inside.
    Pool predictions across folds for honest spatial-out-of-sample metrics.
    """
    import numpy as np
    import polars as pl
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, roc_auc_score

    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet")
    df = df.filter(pl.col("weight") >= min_weight)

    X = df.select(FEATURE_COLS_SHALLOW).to_numpy()
    y = df["label"].to_numpy().astype("int64")
    w = df["weight"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()

    blocks = _spatial_blocks(lat, lon, n_lat=n_lat, n_lon=n_lon)
    unique_blocks = sorted(set(blocks))

    # Spatial leave-one-block-out CV
    fold_results: list[dict] = []
    pred_oof = np.full(len(y), np.nan, dtype="float64")

    for hold_block in unique_blocks:
        train_mask = blocks != hold_block
        test_mask = blocks == hold_block
        if test_mask.sum() == 0:
            continue
        if y[train_mask].sum() < 3 or (1 - y[train_mask]).sum() < 3:
            # Need both classes in train fold
            continue

        # Inner CV for calibration is wasteful with this little data; use Platt (sigmoid) directly.
        base = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=3,
            reg_lambda=0.5,
            verbose=-1,
        )
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
        p = clf.predict_proba(X[test_mask])[:, 1]
        pred_oof[test_mask] = p

        # Per-fold metrics if both classes present in test
        if y[test_mask].sum() > 0 and (1 - y[test_mask]).sum() > 0:
            fold_results.append({
                "block": int(hold_block),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "n_pos_test": int(y[test_mask].sum()),
                "auc": float(roc_auc_score(y[test_mask], p)),
                "brier": float(brier_score_loss(y[test_mask], p)),
            })

    valid = ~np.isnan(pred_oof)
    overall = {
        "n_total": int(len(y)),
        "n_used": int(valid.sum()),
        "n_pos": int(y[valid].sum()),
        "n_neg": int((1 - y[valid]).sum()),
        "spatial_blocks_used": len(fold_results),
        "global_auc": float(roc_auc_score(y[valid], pred_oof[valid])),
        "global_brier": float(brier_score_loss(y[valid], pred_oof[valid])),
    }

    # Baseline: predict mean(y) for everyone
    base_p = np.full(valid.sum(), float(y[valid].mean()))
    overall["baseline_brier_constant"] = float(brier_score_loss(y[valid], base_p))

    # Baseline: neutron-flux-only logistic (the "low flux = ice" heuristic)
    flux = X[valid, FEATURE_COLS_SHALLOW.index("ns_epithermal_cps")]
    flux_pred = 1.0 / (1.0 + np.exp((flux - np.median(flux)) / max(flux.std(), 0.1)))  # crude
    overall["baseline_auc_flux_only"] = float(roc_auc_score(y[valid], flux_pred))

    # Calibration sanity
    pred_bin = np.clip((pred_oof[valid] * 10).astype(int), 0, 9)
    cal_rows = []
    for b in range(10):
        m = pred_bin == b
        if m.sum() >= 3:
            cal_rows.append({
                "bin": [b * 0.1, (b + 1) * 0.1],
                "n": int(m.sum()),
                "predicted_mean": float(pred_oof[valid][m].mean()),
                "actual_pos_rate": float(y[valid][m].mean()),
            })

    return {
        "overall": overall,
        "fold_results": fold_results,
        "calibration_table": cal_rows,
        "feature_cols": FEATURE_COLS_SHALLOW,
        "n_lat_blocks": n_lat,
        "n_lon_blocks": n_lon,
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=600)
def train_final_shallow(min_weight: float = 0.5, n_seeds: int = 5) -> dict:
    """Train final calibrated LightGBM on ALL labels with weight >= min_weight.

    Trains an ensemble of `n_seeds` calibrated models for uncertainty estimation.
    Saves model artifacts as a single pickle on the volume.
    """
    import pickle
    import time

    import numpy as np
    import polars as pl
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV

    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet")
    df = df.filter(pl.col("weight") >= min_weight)

    X = df.select(FEATURE_COLS_SHALLOW).to_numpy()
    y = df["label"].to_numpy().astype("int64")
    w = df["weight"].to_numpy()

    t0 = time.time()
    models = []
    for seed in range(n_seeds):
        base = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_data_in_leaf=3,
            reg_lambda=0.5,
            random_state=seed,
            bagging_fraction=0.8,
            bagging_freq=1,
            feature_fraction=0.85,
            verbose=-1,
        )
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X, y, sample_weight=w)
        models.append(clf)

    artifact = {
        "models": models,
        "feature_cols": FEATURE_COLS_SHALLOW,
        "n_train": int(len(y)),
        "n_pos": int(y.sum()),
        "n_seeds": n_seeds,
    }
    out_path = pathlib.Path(PROCESSED) / "model_shallow.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    volume.commit()

    return {
        "model_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "n_train": int(len(y)),
        "n_pos": int(y.sum()),
        "n_seeds": n_seeds,
        "elapsed_s": round(time.time() - t0, 1),
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1200, memory=8192)
def infer_global_shallow() -> dict:
    """Predict shallow ice probability + ensemble-spread uncertainty + data-density uncertainty
    for every cell of the feature stack. Saves three rasters as numpy arrays.
    """
    import pickle
    import time

    import numpy as np
    import polars as pl
    from scipy.spatial import cKDTree

    t0 = time.time()
    arr_lazy = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    arrs = {k: np.asarray(arr_lazy[k]) for k in FEATURE_COLS_SHALLOW}

    # Stack features into (n_cells, n_features) — LightGBM handles NaN natively
    H, W = arrs[FEATURE_COLS_SHALLOW[0]].shape
    n_cells = H * W
    X = np.stack([arrs[k].reshape(-1) for k in FEATURE_COLS_SHALLOW], axis=1).astype("float32")

    with open(pathlib.Path(PROCESSED) / "model_shallow.pkl", "rb") as f:
        artifact = pickle.load(f)

    # Ensemble predictions, then mean and std
    preds = []
    for m in artifact["models"]:
        p = m.predict_proba(X)[:, 1].astype("float32")
        preds.append(p)
    preds = np.stack(preds, axis=0)  # (n_seeds, n_cells)
    p_mean = preds.mean(axis=0).reshape(H, W)
    p_std = preds.std(axis=0).reshape(H, W)

    # Data-density uncertainty: distance to nearest labeled point on the unit sphere
    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet")
    df = df.filter(pl.col("weight") >= 0.5)
    label_lat = np.deg2rad(df["lat"].to_numpy())
    label_lon = np.deg2rad(df["lon"].to_numpy() % 360.0)
    label_xyz = np.stack([
        np.cos(label_lat) * np.cos(label_lon),
        np.cos(label_lat) * np.sin(label_lon),
        np.sin(label_lat),
    ], axis=1)
    tree = cKDTree(label_xyz)

    # Grid lat/lon
    lat_centers = np.linspace(90.0, -90.0, H, endpoint=False) - (180.0 / H) / 2
    lon_centers = np.linspace(0.0, 360.0, W, endpoint=False) + (360.0 / W) / 2
    lat_g, lon_g = np.meshgrid(np.deg2rad(lat_centers), np.deg2rad(lon_centers), indexing="ij")
    grid_xyz = np.stack([
        (np.cos(lat_g) * np.cos(lon_g)).reshape(-1),
        (np.cos(lat_g) * np.sin(lon_g)).reshape(-1),
        np.sin(lat_g).reshape(-1),
    ], axis=1)
    chord, _ = tree.query(grid_xyz, k=1)
    # great-circle angular distance on unit sphere from chord length
    chord = np.clip(chord, 0.0, 2.0)
    angular = 2.0 * np.arcsin(chord / 2.0)  # radians
    distance_km = (angular * 3396.19).reshape(H, W).astype("float32")

    # Combined uncertainty: model spread + data-density (z-scored sum)
    # Normalize each to roughly [0, 1]: model std typical max ~0.3; distance typical max ~5000 km
    u_model = np.clip(p_std / 0.3, 0.0, 1.0)
    u_data = np.clip(distance_km / 2000.0, 0.0, 1.0)
    uncertainty = 0.5 * u_model + 0.5 * u_data

    out_path = pathlib.Path(PROCESSED) / "shallow_inference.npz"
    np.savez_compressed(
        out_path,
        probability=p_mean.astype("float32"),
        ensemble_std=p_std.astype("float32"),
        distance_to_label_km=distance_km,
        uncertainty=uncertainty.astype("float32"),
    )
    volume.commit()

    valid = np.isfinite(p_mean)
    return {
        "out_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "shape": [H, W],
        "elapsed_s": round(time.time() - t0, 1),
        "prob_stats": {
            "mean": float(p_mean[valid].mean()),
            "p10": float(np.percentile(p_mean[valid], 10)),
            "p50": float(np.percentile(p_mean[valid], 50)),
            "p90": float(np.percentile(p_mean[valid], 90)),
            "frac_above_0.5": float((p_mean[valid] > 0.5).mean()),
            "frac_above_0.8": float((p_mean[valid] > 0.8).mean()),
        },
        "ensemble_std_stats": {
            "median": float(np.median(p_std)),
            "p90": float(np.percentile(p_std, 90)),
        },
        "distance_to_label_km_stats": {
            "median": float(np.median(distance_km)),
            "p90": float(np.percentile(distance_km, 90)),
            "max": float(distance_km.max()),
        },
    }


@app.local_entrypoint()
def model_main():
    import json
    print("=== Train final shallow model ===")
    res = train_final_shallow.remote()
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\n=== Global inference + uncertainty ===")
    res2 = infer_global_shallow.remote()
    print(json.dumps(res2, indent=2))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1800, memory=8192)
def train_swim_soft_shallow(
    n_samples: int = 300_000,
    n_seeds: int = 5,
    val_holdout_blocks: int = 6,
    binarize_threshold: float | None = None,
) -> dict:
    """Train calibrated LightGBM ensemble using SWIM ice-consistency as a soft label.

    Training signal comes from millions of SWIM-coverage pixels (vs the 177 hard
    labels of the previous model). Features EXCLUDE SWIM channels so the model
    has to learn the SWIM-style assessment from non-SWIM physics features
    (MOLA topography, neutron flux). That way it can extrapolate to regions
    SWIM doesn't cover.

    Validates against:
      (a) held-out SWIM spatial blocks (does the model generalize to unseen geography?)
      (b) the 177 hand-curated + Dundas hard labels (does the model agree with ground truth?)
    """
    import pickle
    import time

    import numpy as np
    import polars as pl
    from lightgbm import LGBMRegressor
    from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

    rng = np.random.default_rng(42)

    t0 = time.time()
    arr_lazy = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    arrs = {k: np.asarray(arr_lazy[k]) for k in (FEATURE_COLS_NONSWIM + SWIM_SHALLOW_DEPTH_KEYS)}

    # Build SWIM target: per-pixel max ice consistency across the four shallow
    # SWIM channels. NaN where no SWIM coverage. Clipped to [0, 1].
    swim_layers = [arrs[k] for k in SWIM_SHALLOW_DEPTH_KEYS]
    H, W = swim_layers[0].shape

    target = np.full((H, W), np.nan, dtype="float32")
    coverage = np.zeros((H, W), dtype=bool)
    for layer in swim_layers:
        valid = np.isfinite(layer)
        # Clip per-channel to [0, 1] before merging — SWIM 2.0 c-values can extend
        # slightly outside that range and SWIM 4MIM Ci is published in [0, 1].
        clipped = np.clip(layer, 0.0, 1.0)
        # Merge by max where both are present
        target_now = np.where(valid, np.where(np.isnan(target), clipped, np.maximum(target, clipped)), target)
        target = target_now
        coverage |= valid

    n_coverage = int(coverage.sum())
    target_stats = {
        "coverage_pixels": n_coverage,
        "coverage_fraction": round(n_coverage / (H * W), 4),
        "target_min": float(target[coverage].min()),
        "target_p10": float(np.percentile(target[coverage], 10)),
        "target_p50": float(np.percentile(target[coverage], 50)),
        "target_p90": float(np.percentile(target[coverage], 90)),
        "target_max": float(target[coverage].max()),
        "target_mean": float(target[coverage].mean()),
    }
    print(f"SWIM target stats: {target_stats}")

    # Features (non-SWIM) into (H*W, F)
    X_full = np.stack([arrs[k].reshape(-1) for k in FEATURE_COLS_NONSWIM], axis=1).astype("float32")
    y_full = target.reshape(-1)
    coverage_flat = coverage.reshape(-1)

    coverage_idx = np.where(coverage_flat)[0]
    sample_size = min(n_samples, len(coverage_idx))
    sample_idx = rng.choice(coverage_idx, size=sample_size, replace=False)

    X = X_full[sample_idx]
    y_soft = y_full[sample_idx]
    if binarize_threshold is not None:
        y_soft = (y_soft >= binarize_threshold).astype("float32")

    # Spatial holdout: assign each sampled pixel a 6×6 lat/lon block, hold out N blocks
    sample_rows = sample_idx // W
    sample_cols = sample_idx % W
    sample_lat = 90.0 - (sample_rows + 0.5) * 180.0 / H
    sample_lon = (sample_cols + 0.5) * 360.0 / W
    blocks = _spatial_blocks(sample_lat, sample_lon, n_lat=6, n_lon=6)
    unique_blocks = np.unique(blocks)
    n_val = min(val_holdout_blocks, max(1, len(unique_blocks) // 4))
    val_blocks_chosen = rng.choice(unique_blocks, size=n_val, replace=False)
    val_mask = np.isin(blocks, val_blocks_chosen)
    train_mask = ~val_mask

    print(
        f"Train: {train_mask.sum():,} samples in {len(unique_blocks)-n_val} blocks; "
        f"Val: {val_mask.sum():,} samples in {n_val} held-out blocks"
    )

    # Train ensemble
    models = []
    for seed in range(n_seeds):
        regr = LGBMRegressor(
            objective="cross_entropy",  # log-loss with continuous targets in [0, 1]
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            min_data_in_leaf=64,
            reg_lambda=0.5,
            random_state=seed,
            bagging_fraction=0.8,
            bagging_freq=1,
            feature_fraction=0.85,
            verbose=-1,
        )
        regr.fit(X[train_mask], y_soft[train_mask])
        models.append(regr)

    # Validate on held-out SWIM blocks
    preds_val = np.mean([m.predict(X[val_mask]) for m in models], axis=0)
    preds_val = np.clip(preds_val, 0.0, 1.0)
    val_mae = float(mean_absolute_error(y_soft[val_mask], preds_val))
    val_brier_soft = float(((preds_val - y_soft[val_mask]) ** 2).mean())
    # AUC needs binarized truth; threshold target at 0.5 for the AUC metric only
    y_val_bin = (y_soft[val_mask] >= 0.5).astype("int64")
    if 0 < y_val_bin.sum() < len(y_val_bin):
        val_auc = float(roc_auc_score(y_val_bin, preds_val))
    else:
        val_auc = float("nan")

    # Validate on hard labels
    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet")
    df = df.filter(pl.col("weight") >= 0.5)
    X_hard = df.select(FEATURE_COLS_NONSWIM).to_numpy()
    y_hard = df["label"].to_numpy().astype("int64")
    preds_hard = np.mean([m.predict(X_hard) for m in models], axis=0)
    preds_hard = np.clip(preds_hard, 0.0, 1.0)
    hard_auc = float(roc_auc_score(y_hard, preds_hard)) if 0 < y_hard.sum() < len(y_hard) else float("nan")
    hard_brier = float(brier_score_loss(y_hard, preds_hard))

    artifact = {
        "models": models,
        "feature_cols": FEATURE_COLS_NONSWIM,
        "swim_target_keys": SWIM_SHALLOW_DEPTH_KEYS,
        "n_samples_train": int(train_mask.sum()),
        "n_samples_val": int(val_mask.sum()),
        "n_seeds": n_seeds,
        "binarize_threshold": binarize_threshold,
    }
    out_path = pathlib.Path(PROCESSED) / "model_swim_shallow.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    volume.commit()

    return {
        "model_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "elapsed_s": round(time.time() - t0, 1),
        "target_stats": target_stats,
        "swim_holdout": {
            "mae": round(val_mae, 4),
            "brier_soft": round(val_brier_soft, 4),
            "auc_at_0.5_threshold": round(val_auc, 4),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_held_out_blocks": int(n_val),
        },
        "hard_label_validation": {
            "n_labels": int(len(y_hard)),
            "auc": round(hard_auc, 4),
            "brier": round(hard_brier, 4),
        },
        "feature_importances_mean_gain": {
            k: round(float(np.mean([m.booster_.feature_importance("gain")[i] for m in models])), 1)
            for i, k in enumerate(FEATURE_COLS_NONSWIM)
        },
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1200, memory=8192)
def infer_global_swim_soft() -> dict:
    """Predict shallow ice probability + ensemble-spread uncertainty for every cell
    using the SWIM-soft-label model. Uncertainty is ONLY ensemble disagreement —
    no distance-to-label term — so the visualization doesn't conflate
    'we measured here' with 'the model is unsure here'.
    """
    import pickle
    import time

    import numpy as np

    t0 = time.time()
    arr_lazy = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    arrs = {k: np.asarray(arr_lazy[k]) for k in FEATURE_COLS_NONSWIM}
    H, W = arrs[FEATURE_COLS_NONSWIM[0]].shape
    X = np.stack([arrs[k].reshape(-1) for k in FEATURE_COLS_NONSWIM], axis=1).astype("float32")

    with open(pathlib.Path(PROCESSED) / "model_swim_shallow.pkl", "rb") as f:
        artifact = pickle.load(f)

    preds = []
    for m in artifact["models"]:
        p = np.clip(m.predict(X), 0.0, 1.0).astype("float32")
        preds.append(p)
    preds = np.stack(preds, axis=0)
    p_mean = preds.mean(axis=0).reshape(H, W)
    p_std = preds.std(axis=0).reshape(H, W)

    # Uncertainty = ensemble std, normalized to [0, 1] using the 99th percentile
    # of the actual ensemble std distribution as the ceiling. The SWIM-trained
    # ensemble is internally very consistent (max p_std ~0.12, p90 ~0.015), so
    # a fixed 0.30 ceiling would compress everything into a tiny part of the
    # 8-bit range. Using the empirical p99 keeps the dynamic range usable.
    p99 = float(np.percentile(p_std, 99))
    ceiling = max(p99, 0.02)  # don't go below a sensible minimum
    uncertainty = np.clip(p_std / ceiling, 0.0, 1.0).astype("float32")

    out_path = pathlib.Path(PROCESSED) / "swim_shallow_inference.npz"
    np.savez_compressed(
        out_path,
        probability=p_mean.astype("float32"),
        ensemble_std=p_std.astype("float32"),
        uncertainty=uncertainty,
    )
    volume.commit()

    valid = np.isfinite(p_mean)
    return {
        "out_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "shape": [H, W],
        "elapsed_s": round(time.time() - t0, 1),
        "prob_stats": {
            "mean": float(p_mean[valid].mean()),
            "p10": float(np.percentile(p_mean[valid], 10)),
            "p50": float(np.percentile(p_mean[valid], 50)),
            "p90": float(np.percentile(p_mean[valid], 90)),
            "frac_above_0.5": float((p_mean[valid] > 0.5).mean()),
            "frac_above_0.8": float((p_mean[valid] > 0.8).mean()),
        },
        "ensemble_std_stats": {
            "median": float(np.median(p_std)),
            "p90": float(np.percentile(p_std, 90)),
            "max": float(p_std.max()),
        },
    }


@app.local_entrypoint()
def swim_main():
    import json
    print("=== Train SWIM-soft-label shallow model ===")
    res = train_swim_soft_shallow.remote()
    print(json.dumps(res, indent=2))
    print("\n=== Global inference (SWIM model, ensemble-std uncertainty only) ===")
    res2 = infer_global_swim_soft.remote()
    print(json.dumps(res2, indent=2))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1800, memory=8192)
def train_swim_soft_deep(
    n_samples: int = 300_000,
    n_seeds: int = 5,
    val_holdout_blocks: int = 6,
) -> dict:
    """Mirror of train_swim_soft_shallow but targeting SWIM_DEEP_DEPTH_KEYS (5 m+).

    No hard-label validation — we don't have deep hard labels yet (Stuurman/Petersen
    are paywalled; Bramson PDF on disk needs manual table extraction). Once those
    land, add a deep-label parquet and validate here.
    """
    import pickle
    import time

    import numpy as np
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, roc_auc_score

    rng = np.random.default_rng(42)

    t0 = time.time()
    arr_lazy = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    arrs = {k: np.asarray(arr_lazy[k]) for k in (FEATURE_COLS_NONSWIM + SWIM_DEEP_DEPTH_KEYS)}

    swim_layers = [arrs[k] for k in SWIM_DEEP_DEPTH_KEYS]
    H, W = swim_layers[0].shape

    target = np.full((H, W), np.nan, dtype="float32")
    coverage = np.zeros((H, W), dtype=bool)
    for layer in swim_layers:
        valid = np.isfinite(layer)
        clipped = np.clip(layer, 0.0, 1.0)
        target_now = np.where(valid, np.where(np.isnan(target), clipped, np.maximum(target, clipped)), target)
        target = target_now
        coverage |= valid

    n_coverage = int(coverage.sum())
    target_stats = {
        "coverage_pixels": n_coverage,
        "coverage_fraction": round(n_coverage / (H * W), 4),
        "target_min": float(target[coverage].min()),
        "target_p10": float(np.percentile(target[coverage], 10)),
        "target_p50": float(np.percentile(target[coverage], 50)),
        "target_p90": float(np.percentile(target[coverage], 90)),
        "target_max": float(target[coverage].max()),
        "target_mean": float(target[coverage].mean()),
    }
    print(f"SWIM deep target stats: {target_stats}")

    X_full = np.stack([arrs[k].reshape(-1) for k in FEATURE_COLS_NONSWIM], axis=1).astype("float32")
    y_full = target.reshape(-1)
    coverage_flat = coverage.reshape(-1)

    coverage_idx = np.where(coverage_flat)[0]
    sample_size = min(n_samples, len(coverage_idx))
    sample_idx = rng.choice(coverage_idx, size=sample_size, replace=False)

    X = X_full[sample_idx]
    y_soft = y_full[sample_idx]

    sample_rows = sample_idx // W
    sample_cols = sample_idx % W
    sample_lat = 90.0 - (sample_rows + 0.5) * 180.0 / H
    sample_lon = (sample_cols + 0.5) * 360.0 / W
    blocks = _spatial_blocks(sample_lat, sample_lon, n_lat=6, n_lon=6)
    unique_blocks = np.unique(blocks)
    n_val = min(val_holdout_blocks, max(1, len(unique_blocks) // 4))
    val_blocks_chosen = rng.choice(unique_blocks, size=n_val, replace=False)
    val_mask = np.isin(blocks, val_blocks_chosen)
    train_mask = ~val_mask

    print(
        f"Train: {train_mask.sum():,} samples in {len(unique_blocks)-n_val} blocks; "
        f"Val: {val_mask.sum():,} samples in {n_val} held-out blocks"
    )

    models = []
    for seed in range(n_seeds):
        regr = LGBMRegressor(
            objective="cross_entropy",
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            min_data_in_leaf=64,
            reg_lambda=0.5,
            random_state=seed,
            bagging_fraction=0.8,
            bagging_freq=1,
            feature_fraction=0.85,
            verbose=-1,
        )
        regr.fit(X[train_mask], y_soft[train_mask])
        models.append(regr)

    preds_val = np.mean([m.predict(X[val_mask]) for m in models], axis=0)
    preds_val = np.clip(preds_val, 0.0, 1.0)
    val_mae = float(mean_absolute_error(y_soft[val_mask], preds_val))
    val_brier_soft = float(((preds_val - y_soft[val_mask]) ** 2).mean())
    y_val_bin = (y_soft[val_mask] >= 0.5).astype("int64")
    if 0 < y_val_bin.sum() < len(y_val_bin):
        val_auc = float(roc_auc_score(y_val_bin, preds_val))
    else:
        val_auc = float("nan")

    artifact = {
        "models": models,
        "feature_cols": FEATURE_COLS_NONSWIM,
        "swim_target_keys": SWIM_DEEP_DEPTH_KEYS,
        "n_samples_train": int(train_mask.sum()),
        "n_samples_val": int(val_mask.sum()),
        "n_seeds": n_seeds,
    }
    out_path = pathlib.Path(PROCESSED) / "model_swim_deep.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    volume.commit()

    return {
        "model_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "elapsed_s": round(time.time() - t0, 1),
        "target_stats": target_stats,
        "swim_holdout": {
            "mae": round(val_mae, 4),
            "brier_soft": round(val_brier_soft, 4),
            "auc_at_0.5_threshold": round(val_auc, 4),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_held_out_blocks": int(n_val),
        },
        "feature_importances_mean_gain": {
            k: round(float(np.mean([m.booster_.feature_importance("gain")[i] for m in models])), 1)
            for i, k in enumerate(FEATURE_COLS_NONSWIM)
        },
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1200, memory=8192)
def infer_global_swim_soft_deep() -> dict:
    """Predict deep ice probability + ensemble-spread uncertainty using the deep model."""
    import pickle
    import time

    import numpy as np

    t0 = time.time()
    arr_lazy = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    arrs = {k: np.asarray(arr_lazy[k]) for k in FEATURE_COLS_NONSWIM}
    H, W = arrs[FEATURE_COLS_NONSWIM[0]].shape
    X = np.stack([arrs[k].reshape(-1) for k in FEATURE_COLS_NONSWIM], axis=1).astype("float32")

    with open(pathlib.Path(PROCESSED) / "model_swim_deep.pkl", "rb") as f:
        artifact = pickle.load(f)

    preds = []
    for m in artifact["models"]:
        p = np.clip(m.predict(X), 0.0, 1.0).astype("float32")
        preds.append(p)
    preds = np.stack(preds, axis=0)
    p_mean = preds.mean(axis=0).reshape(H, W)
    p_std = preds.std(axis=0).reshape(H, W)

    p99 = float(np.percentile(p_std, 99))
    ceiling = max(p99, 0.02)
    uncertainty = np.clip(p_std / ceiling, 0.0, 1.0).astype("float32")

    out_path = pathlib.Path(PROCESSED) / "swim_deep_inference.npz"
    np.savez_compressed(
        out_path,
        probability=p_mean.astype("float32"),
        ensemble_std=p_std.astype("float32"),
        uncertainty=uncertainty,
    )
    volume.commit()

    valid = np.isfinite(p_mean)
    return {
        "out_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "shape": [H, W],
        "elapsed_s": round(time.time() - t0, 1),
        "prob_stats": {
            "mean": float(p_mean[valid].mean()),
            "p10": float(np.percentile(p_mean[valid], 10)),
            "p50": float(np.percentile(p_mean[valid], 50)),
            "p90": float(np.percentile(p_mean[valid], 90)),
            "frac_above_0.5": float((p_mean[valid] > 0.5).mean()),
            "frac_above_0.8": float((p_mean[valid] > 0.8).mean()),
        },
        "ensemble_std_stats": {
            "median": float(np.median(p_std)),
            "p90": float(np.percentile(p_std, 90)),
            "max": float(p_std.max()),
        },
    }


@app.local_entrypoint()
def swim_deep_main():
    import json
    print("=== Train SWIM-soft-label deep model (5 m+) ===")
    res = train_swim_soft_deep.remote()
    print(json.dumps(res, indent=2))
    print("\n=== Global deep inference ===")
    res2 = infer_global_swim_soft_deep.remote()
    print(json.dumps(res2, indent=2))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def validate_swim_on_hard_labels() -> dict:
    """Run the deployed SWIM-soft model against the (now Daubar-enriched) hard label set.

    Reports overall AUC/Brier and per-catalog breakdowns so we can see whether the
    model generalizes outside the original 177-label cluster.
    """
    import pickle

    import numpy as np
    import polars as pl
    from sklearn.metrics import brier_score_loss, roc_auc_score

    with open(pathlib.Path(PROCESSED) / "model_swim_shallow.pkl", "rb") as f:
        artifact = pickle.load(f)
    models = artifact["models"]
    feature_cols = artifact["feature_cols"]

    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet")
    df = df.filter(pl.col("weight") >= 0.5)

    # Drop rows where any required feature is NaN — labels at very high latitudes can fall
    # outside Odyssey NS coverage.
    feat_cols_present = [c for c in feature_cols if c in df.columns]
    if len(feat_cols_present) != len(feature_cols):
        return {"error": f"missing features: {set(feature_cols) - set(feat_cols_present)}"}

    valid_mask = df.select(
        [pl.col(c).is_finite() & pl.col(c).is_not_null() for c in feature_cols]
    ).to_numpy().all(axis=1)
    df = df.filter(pl.Series(valid_mask))

    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy().astype("int64")
    preds = np.clip(np.mean([m.predict(X) for m in models], axis=0), 0.0, 1.0)

    def _metrics(mask: np.ndarray) -> dict:
        y_sub = y[mask]
        p_sub = preds[mask]
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "n_pos": 0, "auc": None, "brier": None}
        # Brier is well-defined whenever we have any samples — even single-class.
        brier = round(float(((p_sub - y_sub) ** 2).mean()), 4)
        if 0 < y_sub.sum() < n:
            auc = round(float(roc_auc_score(y_sub, p_sub)), 4)
        else:
            auc = None
        return {"n": n, "n_pos": int(y_sub.sum()), "auc": auc, "brier": brier}

    overall = _metrics(np.ones(len(y), dtype=bool))

    # Per-catalog breakdown: catalog name = first whitespace-separated token of source string
    catalog = df.with_columns(pl.col("source").str.split(" ").list.first().alias("cat"))["cat"].to_numpy()
    by_catalog: dict[str, dict] = {}
    for c in sorted(set(catalog)):
        by_catalog[c] = _metrics(catalog == c)

    # Hemisphere & latitude-band breakdown — was the latitude proxy actually doing real work?
    lat = df["lat"].to_numpy()
    hemis = {
        "north_>60N": _metrics(lat > 60.0),
        "north_30_60N": _metrics((lat > 30.0) & (lat <= 60.0)),
        "tropics_30S_30N": _metrics(np.abs(lat) <= 30.0),
        "south_30_60S": _metrics((lat < -30.0) & (lat >= -60.0)),
        "south_<60S": _metrics(lat < -60.0),
    }

    return {
        "feature_cols": feature_cols,
        "n_total": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "overall": overall,
        "by_catalog": by_catalog,
        "by_latitude_band": hemis,
    }


@app.local_entrypoint()
def validate_main():
    import json
    res = validate_swim_on_hard_labels.remote()
    print(json.dumps(res, indent=2))


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def analyze_inference() -> dict:
    """Lat-band breakdown + feature importance to sanity-check the inference map."""
    import pickle

    import numpy as np

    inf = np.load(pathlib.Path(PROCESSED) / "shallow_inference.npz")
    p = inf["probability"]
    u = inf["uncertainty"]
    H, _ = p.shape

    lat_centers = np.linspace(90.0, -90.0, H, endpoint=False) - (180.0 / H) / 2

    bands = []
    for lo in range(-90, 90, 15):
        mask = (lat_centers >= lo) & (lat_centers < lo + 15)
        bands.append({
            "lat_band": [lo, lo + 15],
            "prob_mean": round(float(p[mask].mean()), 3),
            "prob_p90": round(float(np.percentile(p[mask], 90)), 3),
            "uncertainty_mean": round(float(u[mask].mean()), 3),
        })

    # Feature importance from the first ensemble member
    with open(pathlib.Path(PROCESSED) / "model_shallow.pkl", "rb") as f:
        artifact = pickle.load(f)
    # CalibratedClassifierCV has .calibrated_classifiers_ each with .estimator (the base LightGBM)
    feature_imps: dict[str, list[float]] = {k: [] for k in artifact["feature_cols"]}
    for clf in artifact["models"]:
        for cal in clf.calibrated_classifiers_:
            booster = cal.estimator.booster_
            imps = booster.feature_importance(importance_type="gain")
            for name, imp in zip(artifact["feature_cols"], imps, strict=True):
                feature_imps[name].append(float(imp))
    feature_summary = {
        k: {"mean_gain": round(float(np.mean(v)), 1), "share": round(float(np.mean(v) / max(1.0, sum(np.mean(v2) for v2 in feature_imps.values()))), 3)}
        for k, v in feature_imps.items()
    }
    feature_summary = dict(sorted(feature_summary.items(), key=lambda x: -x[1]["mean_gain"]))

    return {"lat_bands": bands, "feature_importance": feature_summary}


@app.local_entrypoint()
def analyze_main():
    import json
    print(json.dumps(analyze_inference.remote(), indent=2))


# Web tile export. Two PNGs at 4320x2160 + a manifest. deck.gl's BitmapLayer can render
# either directly with bounds=[0, -90, 360, 90] (or [-180, -90, 180, 90] after lon shift).
TILES_DIR = f"{DATA_DIR}/tiles"


@app.function(image=image.pip_install("Pillow>=10"), volumes={DATA_DIR: volume}, timeout=600)
def export_tiles(source: str = "swim") -> dict:
    """Quantize probability + uncertainty to 8-bit PNGs with colormaps applied.
    Also writes a manifest.json the frontend can read for bounds and metadata.

    `source` selects which inference file to export:
      "swim"  → swim_shallow_inference.npz (SWIM-soft-label model, ensemble-std uncertainty)
      "v1"    → shallow_inference.npz (legacy 177-label model with distance-to-label uncertainty)
    """
    import json
    import time

    import numpy as np
    from PIL import Image

    out_dir = pathlib.Path(TILES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "swim":
        inf_path = pathlib.Path(PROCESSED) / "swim_shallow_inference.npz"
        prefix = "shallow"
        depth_label = "shallow_0_5m"
        depth_min, depth_max = 0, 5
    elif source == "swim_deep":
        inf_path = pathlib.Path(PROCESSED) / "swim_deep_inference.npz"
        prefix = "deep"
        depth_label = "deep_5m_plus"
        depth_min, depth_max = 5, 500
    elif source == "v1":
        inf_path = pathlib.Path(PROCESSED) / "shallow_inference.npz"
        prefix = "shallow"
        depth_label = "shallow_0_5m"
        depth_min, depth_max = 0, 5
    else:
        raise ValueError(f"unknown source {source!r}; expected 'swim', 'swim_deep', or 'v1'")
    inf = np.load(inf_path)
    prob = inf["probability"]  # 0..1, may have NaN at unsupported cells (none currently)
    unc = inf["uncertainty"]   # 0..1

    # Roll to lon=[-180, 180] convention that web maps expect (currently 0..360)
    H, W = prob.shape
    half = W // 2
    prob = np.concatenate([prob[:, half:], prob[:, :half]], axis=1)
    unc = np.concatenate([unc[:, half:], unc[:, :half]], axis=1)

    # Replace NaN with sentinel (0 in alpha will mark "no data")
    valid_prob = np.isfinite(prob)
    valid_unc = np.isfinite(unc)

    # Probability colormap: viridis-ish (coded inline so we don't need matplotlib)
    # Map t=[0,1] → RGB via simple linear segments:
    # Real matplotlib LUTs sampled at 9 anchor points then linearly interpolated to 256.
    # Avoids the matplotlib dependency in the Modal image while staying perceptually faithful.
    def _interp_lut(anchors_rgb01: list[tuple[float, float, float]]) -> "np.ndarray":
        anchors = np.asarray(anchors_rgb01, dtype="float32")
        n = anchors.shape[0]
        x_in = np.linspace(0.0, 1.0, n)
        x_out = np.linspace(0.0, 1.0, 256)
        rgb = np.stack([np.interp(x_out, x_in, anchors[:, c]) for c in range(3)], axis=1)
        return np.clip(rgb * 255, 0, 255).astype("uint8")

    def viridis_lut() -> "np.ndarray":
        return _interp_lut([
            (0.267, 0.005, 0.329), (0.283, 0.131, 0.449), (0.254, 0.265, 0.530),
            (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
            (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
            (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
        ])

    def magma_lut() -> "np.ndarray":
        # Real magma anchors (matplotlib): black → purple → magenta → orange → pale yellow
        return _interp_lut([
            (0.001, 0.001, 0.014), (0.072, 0.039, 0.184), (0.184, 0.054, 0.349),
            (0.319, 0.072, 0.413), (0.453, 0.105, 0.430), (0.594, 0.142, 0.415),
            (0.741, 0.198, 0.380), (0.873, 0.288, 0.310), (0.962, 0.426, 0.232),
            (0.995, 0.605, 0.226), (0.998, 0.815, 0.380), (0.987, 0.991, 0.749),
        ])

    prob_lut = viridis_lut()
    unc_lut = magma_lut()

    def colorize(arr: "np.ndarray", lut: "np.ndarray", valid: "np.ndarray") -> "np.ndarray":
        idx = np.clip((arr * 255).astype("int32"), 0, 255)
        rgb = lut[idx]  # (H, W, 3)
        alpha = (valid.astype("uint8")) * 255
        return np.concatenate([rgb, alpha[..., None]], axis=2)

    prob_rgba = colorize(np.where(valid_prob, prob, 0.0), prob_lut, valid_prob)
    unc_rgba = colorize(np.where(valid_unc, unc, 0.0), unc_lut, valid_unc)

    # Also raw quantized 8-bit single-channel for downstream colormapping in JS
    prob_q = np.where(valid_prob, np.clip((prob * 255).round(), 0, 255), 0).astype("uint8")
    unc_q = np.where(valid_unc, np.clip((unc * 255).round(), 0, 255), 0).astype("uint8")

    paths: dict[str, str] = {}
    t0 = time.time()
    for name, arr in [
        (f"{prefix}_probability_rgba.png", prob_rgba),
        (f"{prefix}_uncertainty_rgba.png", unc_rgba),
        (f"{prefix}_probability_8bit.png", prob_q),
        (f"{prefix}_uncertainty_8bit.png", unc_q),
    ]:
        p = out_dir / name
        Image.fromarray(arr).save(p, optimize=True)
        paths[name] = str(p)

    # Per-depth manifest fragment. Frontend can read both shallow and deep manifests
    # to know what depth bin each tile set represents.
    manifest = {
        "version": "v1",
        "depth_bin": {depth_label: {"min_m": depth_min, "max_m": depth_max}},
        "crs": TARGET_CRS,
        "shape": [H, W],
        "bounds_lonlat": [-180.0, -90.0, 180.0, 90.0],
        "files": {k: pathlib.Path(v).name for k, v in paths.items()},
        "encoding": {
            "rgba": "viridis (probability) / magma (uncertainty), alpha=0 means no data",
            "8bit": "uint8 single channel, value = (probability or uncertainty) * 255, 0 if no data",
        },
        "model": {
            "name": f"mars-ice {prefix}",
            "source": source,
        },
    }
    manifest_name = "manifest.json" if prefix == "shallow" else f"manifest_{prefix}.json"
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    paths[manifest_name] = str(out_dir / manifest_name)

    volume.commit()
    sizes = {pathlib.Path(p).name: round(pathlib.Path(p).stat().st_size / 1024 / 1024, 3) for p in paths.values()}
    return {
        "out_dir": str(out_dir),
        "files": paths,
        "sizes_mb": sizes,
        "elapsed_s": round(time.time() - t0, 1),
        "total_mb": round(sum(sizes.values()), 3),
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def get_tile_bytes(filename: str) -> bytes:
    """Return raw bytes of a tile file for local download."""
    p = pathlib.Path(TILES_DIR) / filename
    return p.read_bytes()


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def export_labels_compact() -> list[dict]:
    """Export training labels as a small JSON-friendly list of dicts.
    Only the high-confidence subset (weight >= 0.5) — same filter the final model uses.
    """
    import polars as pl

    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow.parquet")
    df = df.filter(pl.col("weight") >= 0.5)
    rows: list[dict] = []
    for r in df.iter_rows(named=True):
        rows.append(
            {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "label": float(r["label"]),
                "weight": float(r["weight"]),
                "source": str(r.get("source", "unknown")),
            }
        )
    return rows


@app.local_entrypoint()
def fetch_labels(out_path: str = "web/labels.json"):
    import json
    import pathlib as plib

    rows = export_labels_compact.remote()
    target = plib.Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows))
    print(f"  wrote {len(rows)} labels -> {target}")


@app.local_entrypoint()
def export_main():
    import json
    res = export_tiles.remote()
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=2))


@app.local_entrypoint()
def fetch_tiles(out_dir: str = "tiles_local", include_deep: bool = True):
    """Download the exported tiles to a local directory for preview."""
    import pathlib as plib

    target = plib.Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    names = [
        "shallow_probability_rgba.png",
        "shallow_uncertainty_rgba.png",
        "shallow_probability_8bit.png",
        "shallow_uncertainty_8bit.png",
        "manifest.json",
    ]
    if include_deep:
        names += [
            "deep_probability_rgba.png",
            "deep_uncertainty_rgba.png",
            "deep_probability_8bit.png",
            "deep_uncertainty_8bit.png",
            "manifest_deep.json",
        ]
    for name in names:
        try:
            b = get_tile_bytes.remote(name)
            (target / name).write_bytes(b)
            print(f"  {name}: {len(b)} bytes -> {target / name}")
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:120]}")


@app.local_entrypoint()
def train_main():
    import json
    res = train_shallow_baseline.remote()
    print("OVERALL:")
    for k, v in res["overall"].items():
        print(f"  {k}: {v}")
    print("\nPER-FOLD (held-out spatial block):")
    for f in res["fold_results"]:
        print(f"  block={f['block']:<3}  n_train={f['n_train']:>3}  n_test={f['n_test']:>3}  "
              f"n_pos={f['n_pos_test']:>2}  AUC={f['auc']:.3f}  Brier={f['brier']:.3f}")
    print("\nCALIBRATION (predicted prob vs actual pos rate, 10 bins):")
    for c in res["calibration_table"]:
        print(f"  [{c['bin'][0]:.1f},{c['bin'][1]:.1f})  n={c['n']:>3}  "
              f"pred_mean={c['predicted_mean']:.2f}  actual={c['actual_pos_rate']:.2f}")


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def dundas_distribution() -> dict:
    """Return TableS1 status counts, TableS2 status counts, TableS4 yes/no counts."""
    import polars as pl

    out: dict = {}
    base = pathlib.Path(RAW) / "labels/dundas_2021"

    s1 = pl.read_csv(base / "TableS1_final.csv")
    out["TableS1_status_counts"] = dict(s1.group_by("Status").agg(pl.len().alias("n")).iter_rows())

    s2 = pl.read_csv(base / "TableS2_final.csv")
    out["TableS2_status_counts"] = dict(s2.group_by("Status").agg(pl.len().alias("n")).iter_rows())

    s4 = pl.read_csv(base / "TableS4_final.csv")
    out["TableS4_ice_counts"] = dict(s4.group_by("Ice exposure?").agg(pl.len().alias("n")).iter_rows())
    out["TableS4_lat_range"] = [float(s4["Latitude"].min()), float(s4["Latitude"].max())]
    out["TableS4_lon_range"] = [float(s4["Longitude"].min()), float(s4["Longitude"].max())]

    out["TableS1_lat_range"] = [float(s1["Latitude"].min()), float(s1["Latitude"].max())]
    out["TableS1_lon_range"] = [float(s1["Longitude"].min()), float(s1["Longitude"].max())]

    return out


@app.local_entrypoint()
def dundas_dist_main():
    import json
    print(json.dumps(dundas_distribution.remote(), indent=2))


# Target grid for the unified feature stack:
#   IAU_2015:49900 geographic (ocentric, east-positive 0..360 lon, -90..90 lat)
#   4320 cols * 2160 rows = 1/12 degree = ~4.94 km/pixel at equator
TARGET_CRS = "IAU_2015:49900"
TARGET_WIDTH = 4320
TARGET_HEIGHT = 2160
TARGET_LON_MIN, TARGET_LON_MAX = 0.0, 360.0  # east-positive
TARGET_LAT_MIN, TARGET_LAT_MAX = -90.0, 90.0
PROCESSED = f"{DATA_DIR}/processed"


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=1800)
def build_feature_stack() -> dict:
    """Regrid every input layer to a unified 4320x2160 lat/lon grid on IAU_2015:49900.

    Saves a single .npz with named arrays + a metadata JSON. Returns summary stats.
    """
    import json

    import numpy as np
    import polars as pl
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    out_dir = pathlib.Path(PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_transform = from_bounds(
        TARGET_LON_MIN, TARGET_LAT_MIN, TARGET_LON_MAX, TARGET_LAT_MAX,
        TARGET_WIDTH, TARGET_HEIGHT,
    )

    layers: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    def _regrid_raster(src_path: pathlib.Path, label: str, resampling=Resampling.bilinear) -> np.ndarray:
        with rasterio.open(src_path) as ds:
            src_arr = ds.read(1, masked=True).astype("float32")
            src_crs = ds.crs
            src_transform = ds.transform
            src_nodata = ds.nodata
        out = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype="float32")
        # Replace nodata with NaN for clean reprojection
        if src_nodata is not None:
            src_arr = np.where(src_arr.data == src_nodata, np.nan, src_arr.data).astype("float32")
        else:
            src_arr = np.asarray(src_arr.filled(np.nan), dtype="float32")
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        finite = np.isfinite(out)
        meta[label] = {
            "source": str(src_path.relative_to(pathlib.Path(DATA_DIR))),
            "src_crs": str(src_crs),
            "src_shape": list(src_arr.shape),
            "valid_frac": float(finite.mean()),
            "min": float(np.nanmin(out)) if finite.any() else None,
            "max": float(np.nanmax(out)) if finite.any() else None,
            "mean": float(np.nanmean(out)) if finite.any() else None,
        }
        return out

    def _regrid_table(tab_path: pathlib.Path, label: str, nodata_val: float | None = None) -> np.ndarray:
        df = pl.read_csv(
            tab_path,
            separator=",",
            has_header=False,
            new_columns=["lon", "lat", "value"],
        ).with_columns(pl.col(c).str.strip_chars().cast(pl.Float64) for c in ["lon", "lat", "value"])
        # Source grid: 720x360 at 0.5°, lon east-positive 0..360, lat -89.75..89.75
        src = df.sort(["lat", "lon"])["value"].to_numpy().astype("float32").reshape(360, 720)
        src = src[::-1]  # row 0 = north
        if nodata_val is not None:
            src = np.where(src == nodata_val, np.nan, src)
        # Even the epithermal file uses 0.0 as a stealth mask for far-south cells; treat zero as nodata
        # ONLY in polar bands where flux of literal zero is non-physical.
        src = np.where(src == 0.0, np.nan, src)
        src_transform = from_bounds(0.0, -90.0, 360.0, 90.0, 720, 360)
        out = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype="float32")
        reproject(
            source=src,
            destination=out,
            src_transform=src_transform,
            src_crs=TARGET_CRS,  # already in target geographic CRS
            dst_transform=target_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        finite = np.isfinite(out)
        meta[label] = {
            "source": str(tab_path.relative_to(pathlib.Path(DATA_DIR))),
            "src_shape": [360, 720],
            "valid_frac": float(finite.mean()),
            "min": float(np.nanmin(out)) if finite.any() else None,
            "max": float(np.nanmax(out)) if finite.any() else None,
            "mean": float(np.nanmean(out)) if finite.any() else None,
        }
        return out

    raw = pathlib.Path(RAW)

    # Topography
    layers["mola_topography_m"] = _regrid_raster(raw / "mola_megdr/megt90n000eb.lbl", "mola_topography_m")
    layers["mola_areoid_m"] = _regrid_raster(raw / "mola_megdr/mega90n000eb.lbl", "mola_areoid_m")

    # SWIM ice consistency (mid-lat, multiple depth bins)
    layers["swim4mim_ci_0_1m"] = _regrid_raster(raw / "swim4mim/SWIM4MIM_Ci_0_1.tif", "swim4mim_ci_0_1m")
    layers["swim4mim_ci_1_5m"] = _regrid_raster(raw / "swim4mim/SWIM4MIM_Ci_1_5.tif", "swim4mim_ci_1_5m")
    layers["swim4mim_ci_5m"] = _regrid_raster(raw / "swim4mim/SWIM4MIM_Ci_5.tif", "swim4mim_ci_5m")

    # SWIM 2.0 (extends polar coverage)
    layers["swim2_c_0_1m"] = _regrid_raster(raw / "swim2/SWIM2_c0_1.tif", "swim2_c_0_1m")
    layers["swim2_c_1_5m"] = _regrid_raster(raw / "swim2/SWIM2_c1_5.tif", "swim2_c_1_5m")
    layers["swim2_c_5m"] = _regrid_raster(raw / "swim2/SWIM2_c_5.tif", "swim2_c_5m")

    # Neutron flux (Feldman 2002 first-25-day map; placeholder for production WEH)
    layers["ns_epithermal_cps"] = _regrid_table(raw / "odyssey_grs/ns_epithermal_020917.tab", "ns_epithermal_cps")
    layers["ns_thermal_cps"] = _regrid_table(raw / "odyssey_grs/ns_thermal_020917.tab", "ns_thermal_cps", nodata_val=-99.999)
    layers["ns_fast_cps"] = _regrid_table(raw / "odyssey_grs/ns_fast_020917.tab", "ns_fast_cps", nodata_val=-99.999)

    out_path = out_dir / "feature_stack.npz"
    np.savez_compressed(out_path, **layers)
    meta_path = out_dir / "feature_stack.json"
    meta_path.write_text(json.dumps({
        "target_crs": TARGET_CRS,
        "target_shape": [TARGET_HEIGHT, TARGET_WIDTH],
        "target_bounds": [TARGET_LON_MIN, TARGET_LAT_MIN, TARGET_LON_MAX, TARGET_LAT_MAX],
        "pixel_deg": [(TARGET_LON_MAX - TARGET_LON_MIN) / TARGET_WIDTH, (TARGET_LAT_MAX - TARGET_LAT_MIN) / TARGET_HEIGHT],
        "layers": meta,
    }, indent=2))

    volume.commit()
    return {
        "out_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
        "n_layers": len(layers),
        "layers": list(layers.keys()),
        "meta_summary": {k: {"valid_frac": round(v["valid_frac"], 3), "mean": v["mean"]} for k, v in meta.items()},
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def alignment_check() -> dict:
    """Quantify whether MOLA topography and Odyssey neutron flux co-register.

    Test: in the south polar region (-90..-75 latitude), do the cells with the lowest
    epithermal-neutron flux (ice signature) coincide spatially with elevated topography
    (the south polar layered deposits)? Reports IoU between top decile of (1/flux) and
    top decile of elevation, restricted to the polar zone.
    """
    import numpy as np

    arr = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    topo = arr["mola_topography_m"]
    flux = arr["ns_epithermal_cps"]

    # build lat for each row of the target grid: row 0 = +90, row TARGET_HEIGHT-1 = -90
    lat = np.linspace(90.0, -90.0, TARGET_HEIGHT, endpoint=False) - (180.0 / TARGET_HEIGHT) / 2

    def report_zone(name: str, zone_mask: np.ndarray) -> dict:
        f = flux[zone_mask]
        t = topo[zone_mask]
        valid = np.isfinite(f) & np.isfinite(t)
        if valid.sum() < 100:
            return {"zone": name, "valid_cells": int(valid.sum()), "skipped": True}
        f, t = f[valid], t[valid]
        # ice-signature cells = lowest 10% flux (anomalously low → high H)
        thresh_f = np.percentile(f, 10)
        ice_mask = f <= thresh_f
        # high-topography cells = top 10% elevation
        thresh_t = np.percentile(t, 90)
        high_mask = t >= thresh_t
        # IoU of those two binary masks
        inter = (ice_mask & high_mask).sum()
        union = (ice_mask | high_mask).sum()
        iou = inter / union if union else 0.0
        # Random baseline: 0.10 * 0.10 / (0.10 + 0.10 - 0.01) = 0.0526
        return {
            "zone": name,
            "valid_cells": int(valid.sum()),
            "low_flux_threshold": float(thresh_f),
            "high_topo_threshold_m": float(thresh_t),
            "iou_low_flux_x_high_topo": round(iou, 3),
            "random_baseline_iou": 0.053,
            "lift": round(iou / 0.0526, 2) if iou else 0.0,
        }

    south_polar = np.broadcast_to(lat[:, None] < -75.0, topo.shape)
    north_polar = np.broadcast_to(lat[:, None] > 75.0, topo.shape)
    midlat = np.broadcast_to((lat[:, None] >= -60.0) & (lat[:, None] <= 60.0), topo.shape)

    return {
        "south_polar_below_-75": report_zone("south_polar", south_polar),
        "north_polar_above_+75": report_zone("north_polar", north_polar),
        "midlat_-60_to_+60": report_zone("midlat", midlat),
    }


@app.local_entrypoint()
def stack_main():
    res = build_feature_stack.remote()
    print("STACK:")
    for k, v in res.items():
        if k == "meta_summary":
            print(f"  {k}:")
            for layer, stats in v.items():
                print(f"    {layer:<24} valid_frac={stats['valid_frac']:.3f} mean={stats['mean']}")
        else:
            print(f"  {k}: {v}")


@app.local_entrypoint()
def align_main():
    res = alignment_check.remote()
    for zone, stats in res.items():
        print(f"\n{zone}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def diagnose_alignment() -> dict:
    """Direct check: does the reprojected neutron flux land where we expect it?
    Reports global flux extrema location, polar-zone topography histogram, and
    correlation between flux and topography in mid-latitudes (where neutron data
    isn't dominated by seasonal CO2 frost and topography variation is real).
    """
    import numpy as np

    arr = np.load(pathlib.Path(PROCESSED) / "feature_stack.npz")
    topo = arr["mola_topography_m"]
    flux = arr["ns_epithermal_cps"]
    lat_centers = np.linspace(90, -90, TARGET_HEIGHT, endpoint=False) - (180.0 / TARGET_HEIGHT) / 2
    lon_centers = np.linspace(0, 360, TARGET_WIDTH, endpoint=False) + (360.0 / TARGET_WIDTH) / 2

    # Where is the global flux minimum? (Should be inside the south polar layered deposits.)
    fmask = np.isfinite(flux)
    f = np.where(fmask, flux, np.inf)
    rmin, cmin = np.unravel_index(np.argmin(f), f.shape)
    rmax, cmax = np.unravel_index(np.argmax(np.where(fmask, flux, -np.inf)), f.shape)

    # Where is the highest MOLA topography globally?
    tmask = np.isfinite(topo)
    t = np.where(tmask, topo, -np.inf)
    rt_max, ct_max = np.unravel_index(np.argmax(t), t.shape)

    # Mid-latitude flux–topography correlation (where physics is unambiguous)
    midlat_rows = (lat_centers >= -45) & (lat_centers <= 45)
    f_ml = flux[midlat_rows]
    t_ml = topo[midlat_rows]
    valid = np.isfinite(f_ml) & np.isfinite(t_ml)
    pearson = float(np.corrcoef(f_ml[valid], t_ml[valid])[0, 1])

    # Polar zone flux-mean by lat band (10° bins from -90 to +90)
    band_summary = []
    for lo in range(-90, 90, 10):
        mask = (lat_centers >= lo) & (lat_centers < lo + 10)
        f_band = flux[mask]
        v = f_band[np.isfinite(f_band)]
        band_summary.append({
            "lat_band": [lo, lo + 10],
            "flux_mean": round(float(v.mean()), 3) if v.size else None,
            "flux_p10": round(float(np.percentile(v, 10)), 3) if v.size else None,
        })

    return {
        "global_flux_min": {
            "value": round(float(flux[rmin, cmin]), 3),
            "lat": round(float(lat_centers[rmin]), 2),
            "lon_east": round(float(lon_centers[cmin]), 2),
        },
        "global_flux_max": {
            "value": round(float(flux[rmax, cmax]), 3),
            "lat": round(float(lat_centers[rmax]), 2),
            "lon_east": round(float(lon_centers[cmax]), 2),
        },
        "global_topo_max": {
            "value_m": round(float(topo[rt_max, ct_max]), 0),
            "lat": round(float(lat_centers[rt_max]), 2),
            "lon_east": round(float(lon_centers[ct_max]), 2),
        },
        "midlat_flux_topo_pearson": round(pearson, 3),
        "lat_band_flux_summary": band_summary,
    }


@app.local_entrypoint()
def diag_main():
    import json
    print(json.dumps(diagnose_alignment.remote(), indent=2))


# --- Labels --------------------------------------------------------------------------------------
#
# Two label sets, kept separate per spec:
#   shallow:  depth 0-5 m,   each row is one (lat, lon, label, weight, source, evidence)
#   deep:     depth 5-500 m, each row adds a depth_m field
#
# Label values: 1.0 = high-confidence ice, 0.0 = high-confidence non-ice, with `weight`
# capturing soft confidence. SWIM consistency scores can be folded in later as soft labels.

# Hand-curated ground-truth points from lander/rover missions and well-known sites.
# Coordinates are areocentric, lon east-positive 0..360. These are verifiable from any
# planetary-science textbook; they're the floor of the label set.
GROUND_TRUTH_POINTS: list[dict] = [
    # --- Confirmed shallow ice ---
    {
        "name": "phoenix_landing",
        "lat": 68.22,
        "lon": 234.25,
        "label": 1.0,
        "weight": 1.0,
        "depth_m": 0.05,
        "source": "Phoenix Robotic Arm (Smith 2009 Science 325:58-61); ice exposed by trenching at ~5 cm",
        "evidence": "in_situ_excavation",
    },
    # --- Confirmed dry / non-ice surface near-equator ---
    {
        "name": "viking_1",
        "lat": 22.27,
        "lon": 312.05,
        "label": 0.0,
        "weight": 0.8,
        "depth_m": 0.05,
        "source": "Viking 1 lander (Chryse Planitia); no ice in surface trenching",
        "evidence": "in_situ",
    },
    {
        "name": "pathfinder",
        "lat": 19.13,
        "lon": 326.84,
        "label": 0.0,
        "weight": 0.7,
        "depth_m": 0.1,
        "source": "Mars Pathfinder (Ares Vallis); equatorial dry context",
        "evidence": "in_situ",
    },
    {
        "name": "spirit",
        "lat": -14.57,
        "lon": 175.47,
        "label": 0.0,
        "weight": 0.7,
        "depth_m": 0.1,
        "source": "MER Spirit (Gusev Crater); equatorial; dry near surface",
        "evidence": "in_situ",
    },
    {
        "name": "opportunity",
        "lat": -1.95,
        "lon": 354.47,
        "label": 0.0,
        "weight": 0.7,
        "depth_m": 0.1,
        "source": "MER Opportunity (Meridiani Planum); equatorial; dry near surface",
        "evidence": "in_situ",
    },
    {
        "name": "curiosity_gale",
        "lat": -4.59,
        "lon": 137.44,
        "label": 0.0,
        "weight": 0.6,
        "depth_m": 0.1,
        "source": "MSL Curiosity (Gale Crater); equatorial; near-surface dry per DAN+SAM",
        "evidence": "in_situ",
    },
    {
        "name": "perseverance_jezero",
        "lat": 18.44,
        "lon": 77.45,
        "label": 0.0,
        "weight": 0.5,
        "depth_m": 0.1,
        "source": "Perseverance (Jezero); equatorial; RIMFAX shows little near-surface H2O",
        "evidence": "in_situ_radar",
    },
    # --- Note: Viking 2 (Utopia Planitia) is NOT a clean negative; the area has shallow ice
    # signatures within ~1 m per Stuurman 2016. Excluded deliberately.
]


def _parse_dundas_2021(label_root: pathlib.Path) -> "pl.DataFrame":
    """Combine Dundas 2021 TableS1 (scarps) + TableS4 (craters) into our shallow-label schema."""
    import polars as pl

    base = label_root / "dundas_2021"
    if not base.exists():
        return pl.DataFrame()

    rows: list[dict] = []

    s1 = pl.read_csv(base / "TableS1_final.csv")
    # TableS1: scarp sites, all positive ice exposures (confidence varies by Status)
    status_to_weight = {"C": 1.0, "CM": 1.0, "P": 0.5, "L": 0.5}
    for r in s1.iter_rows(named=True):
        w = status_to_weight.get(r["Status"], 0.4)
        rows.append({
            "name": f"dundas_2021_{r['Site ID']}",
            "lat": float(r["Latitude"]),
            "lon": float(r["Longitude"]) % 360.0,
            "label": 1.0,
            "weight": w,
            "depth_m": 0.05,
            "source": f"Dundas 2021 TableS1 ({r['Status']})",
            "evidence": "scarp_exposure",
        })

    s4 = pl.read_csv(base / "TableS4_final.csv")
    # TableS4: craters with Yes/No/Probable/Possible labels
    ice_to_label_weight = {
        "Yes": (1.0, 1.0),
        "Yes (CTX)": (1.0, 0.9),
        "Probable": (1.0, 0.7),
        "Possible": (1.0, 0.4),
        "No": (0.0, 1.0),
    }
    for r in s4.iter_rows(named=True):
        ice = r["Ice exposure?"]
        if ice not in ice_to_label_weight:
            continue
        label, weight = ice_to_label_weight[ice]
        rows.append({
            "name": f"dundas_2021_{r['Site ID']}",
            "lat": float(r["Latitude"]),
            "lon": float(r["Longitude"]) % 360.0,
            "label": label,
            "weight": weight,
            "depth_m": 0.1,
            "source": f"Dundas 2021 TableS4 ({ice})",
            "evidence": "fresh_impact_crater",
        })

    return pl.DataFrame(rows)


def _parse_daubar_2022(label_root: pathlib.Path) -> "pl.DataFrame":
    """Daubar 2022 catalog of dated impacts → globally distributed hard shallow labels.

    1,203 craters with surface morphology assessed for ice exposure. We turn the
    Ice-exposing impact column into binary labels, using crater diameter as a rough
    proxy for excavation depth (~0.1 × diameter). Most craters are <30 m diameter, so
    excavation reaches ~3 m — appropriate ground truth for the shallow (0–5 m) model.
    """
    import polars as pl

    xlsx = label_root / "daubar_2022/daubar_2022_tableS1.xlsx"
    if not xlsx.exists():
        return pl.DataFrame()

    df = pl.read_excel(xlsx)
    # Column names in the upstream sheet are slightly mislabeled — values are correct.
    lat_col = "Latitude (deg E, centric)"
    lon_col = "Longitude (deg N)"
    ice_col = "Ice-exposing impact"
    diam_col = "Diameter (m)"

    rows: list[dict] = []
    for r in df.iter_rows(named=True):
        try:
            lat = float(r[lat_col])
            lon = float(r[lon_col]) % 360.0
            diam = float(r[diam_col]) if r[diam_col] is not None else None
        except (TypeError, ValueError):
            continue
        ice_raw = (r[ice_col] or "").strip().lower()
        # Excavation depth ≈ 0.1 × diameter; cap at 5 m so this stays a shallow-model label
        if diam is None:
            depth_m = 1.0
        else:
            depth_m = max(0.1, min(5.0, 0.1 * diam))

        if ice_raw == "y":
            label, weight = 1.0, 1.0
        elif ice_raw == "possible":
            label, weight = 1.0, 0.4
        elif ice_raw == "n":
            # "no ice exposed" is strong evidence of no ice within the excavation depth.
            # Weight slightly below 1.0 because the crater might just have missed an ice patch.
            label, weight = 0.0, 0.7
        else:
            continue

        rows.append({
            "name": f"daubar_2022_{r['HiRISE Observation ID']}",
            "lat": lat,
            "lon": lon,
            "label": label,
            "weight": weight,
            "depth_m": depth_m,
            "source": f"Daubar 2022 (ice={ice_raw}, diam={diam:.1f} m)" if diam else f"Daubar 2022 (ice={ice_raw})",
            "evidence": "fresh_impact_crater",
        })

    return pl.DataFrame(rows) if rows else pl.DataFrame()


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def build_labels(extra_shallow_csv: str | None = None, extra_deep_csv: str | None = None) -> dict:
    """Assemble shallow + deep label tables. Output: two parquet files in processed/."""
    import json

    import polars as pl

    out_dir = pathlib.Path(PROCESSED)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hand-curated ground truth → shallow table
    gt_df = pl.DataFrame(GROUND_TRUTH_POINTS)

    # Aggregated catalogs from raw/labels/
    label_root = pathlib.Path(RAW) / "labels"
    sources: list[pl.DataFrame] = [gt_df]
    dundas_df = _parse_dundas_2021(label_root)
    if not dundas_df.is_empty():
        sources.append(dundas_df)
    daubar_df = _parse_daubar_2022(label_root)
    if not daubar_df.is_empty():
        sources.append(daubar_df)

    shallow = pl.concat(sources, how="diagonal_relaxed")
    deep = pl.DataFrame(schema=gt_df.schema)

    shallow_path = out_dir / "labels_shallow.parquet"
    deep_path = out_dir / "labels_deep.parquet"
    shallow.write_parquet(shallow_path)
    deep.write_parquet(deep_path)
    volume.commit()

    pos_strict = ((shallow["label"] == 1.0) & (shallow["weight"] >= 0.7)).sum()
    neg_strict = ((shallow["label"] == 0.0) & (shallow["weight"] >= 0.7)).sum()
    return {
        "shallow_path": str(shallow_path),
        "shallow_count": shallow.height,
        "shallow_pos_total": int((shallow["label"] == 1.0).sum()),
        "shallow_neg_total": int((shallow["label"] == 0.0).sum()),
        "shallow_pos_high_conf": int(pos_strict),
        "shallow_neg_high_conf": int(neg_strict),
        "shallow_lat_range": [float(shallow["lat"].min()), float(shallow["lat"].max())],
        "by_catalog": dict(
            shallow.with_columns(pl.col("source").str.split(" ").list.first().alias("catalog"))
            .group_by("catalog")
            .agg(pl.len().alias("n"))
            .iter_rows()
        ),
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def attach_features_to_labels() -> dict:
    """For each shallow label row, look up the feature-stack values at its (lat, lon).
    Vectorized — loads each feature array once, batch-indexes by precomputed (row, col).
    """
    import numpy as np
    import polars as pl

    arr_path = pathlib.Path(PROCESSED) / "feature_stack.npz"
    arr_lazy = np.load(arr_path)
    layer_keys = list(arr_lazy.files)
    # Force-load each layer once (NpzFile decompresses lazily on each access)
    arrs = {k: np.asarray(arr_lazy[k]) for k in layer_keys}

    shallow = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow.parquet")
    lat = shallow["lat"].to_numpy()
    lon = shallow["lon"].to_numpy() % 360.0
    rows = np.clip(((90.0 - lat) / 180.0 * TARGET_HEIGHT).astype("int64"), 0, TARGET_HEIGHT - 1)
    cols = np.clip((lon / 360.0 * TARGET_WIDTH).astype("int64"), 0, TARGET_WIDTH - 1)

    out_cols: dict[str, np.ndarray] = {}
    for k in layer_keys:
        v = arrs[k][rows, cols]
        out_cols[k] = np.where(np.isfinite(v), v, np.nan)  # NaN where feature is undefined

    enriched = shallow.with_columns(
        *[pl.Series(name=k, values=out_cols[k]) for k in layer_keys]
    )
    out_path = pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet"
    enriched.write_parquet(out_path)
    volume.commit()

    # Compute coverage stats per feature (how often is the feature defined for our labels)
    coverage = {k: float(np.isfinite(out_cols[k]).mean()) for k in layer_keys}
    return {
        "out_path": str(out_path),
        "rows": enriched.height,
        "n_layers": len(layer_keys),
        "coverage": {k: round(v, 3) for k, v in coverage.items()},
    }


@app.local_entrypoint()
def labels_main():
    res = build_labels.remote()
    print("BUILD LABELS:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\nATTACH FEATURES:")
    res2 = attach_features_to_labels.remote()
    print(f"  rows: {res2['rows']}, layers: {res2['n_layers']}")
    print(f"  coverage (frac of labels where feature is defined):")
    for k, v in res2["coverage"].items():
        print(f"    {k:<22} {v:.3f}")


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=120)
def describe_labels() -> str:
    """Print a tabular view of every labeled point with key features for sanity check."""
    import polars as pl

    df = pl.read_parquet(pathlib.Path(PROCESSED) / "labels_shallow_with_features.parquet")
    df = df.with_columns(
        pl.col("mola_topography_m").round(0).cast(pl.Int64).alias("topo_m"),
        pl.col("ns_epithermal_cps").round(2).alias("epi_cps"),
        pl.col("ns_thermal_cps").round(2).alias("thr_cps"),
        pl.col("swim4mim_ci_0_1m").round(2).alias("swim_0_1"),
    ).select(["name", "lat", "lon", "label", "topo_m", "epi_cps", "thr_cps", "swim_0_1", "evidence"])

    rows = df.to_dicts()
    headers = ["name", "lat", "lon", "label", "topo_m", "epi_cps", "thr_cps", "swim_0_1", "evidence"]
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    lines = [line, "  ".join("-" * widths[h] for h in headers)]
    for r in rows:
        lines.append("  ".join(str(r[h]).ljust(widths[h]) for h in headers))
    return "\n".join(lines)


@app.local_entrypoint()
def describe_main():
    print(describe_labels.remote())
