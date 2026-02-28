#!/usr/bin/env python3
"""
Data Cleaning/clean_and_compile.py

Cleans and compiles the master airline ticket dataset (source of truth) and
LEFT-JOINs external datasets WITHOUT changing the original route locations
or the Year/quarter timeline.

Folder structure assumed (your screenshot):
SDSS-DATATHON-1/
  Data Cleaning/
    clean_and_compile.py   <-- this file
  Raw Data/
    Airline/
      airline_ticket_dataset.csv
    JetFuel/
      WJFUELUSGULF.csv
    LabourCost/
      CIS2024300000000I.csv
    Tourism/
      2024-Top-States-and-Cities-Visited.csv
    (optional)
    Aircraft_Landing_Facilities.csv  (if you put it in Raw Data/)

Outputs:
  Data Cleaning/Processed/final_dataset.csv
  (Parquet optional if pyarrow/fastparquet installed)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------- helpers ----------
_STATE_TO_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
    "Delaware":"DE","District of Columbia":"DC","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA",
    "Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
    "Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
    "Wisconsin":"WI","Wyoming":"WY",
    "Hawaiian Islands":"HI",
    "Guam":"GU","Puerto Rico":"PR","U.S. Virgin Islands":"VI","Northern Mariana Islands":"MP","American Samoa":"AS"
}

def parse_money(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("$", "").replace(",", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def parse_number(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def extract_city_state(city_field: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Example: 'Miami, FL (Metropolitan Area)' -> ('Miami', 'FL')
    """
    if pd.isna(city_field):
        return None, None
    s = str(city_field).strip()
    s2 = re.sub(r"\s*\(.*\)\s*$", "", s)  # remove trailing parentheses
    m = re.match(r"^(.*?),\s*([A-Z]{2})\s*$", s2)
    if not m:
        return s2, None
    return m.group(1).strip(), m.group(2).strip()

def add_year_quarter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["Year"] = d[date_col].dt.year.astype("Int64")
    d["quarter"] = d[date_col].dt.quarter.astype("Int64")
    return d


# ---------- master tickets ----------
def load_clean_tickets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = ["Year", "quarter", "citymarketid_1", "citymarketid_2", "city1", "city2",
                "nsmiles", "passengers", "fare"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tickets file missing required columns: {missing}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")

    df["nsmiles"] = pd.to_numeric(df["nsmiles"], errors="coerce")
    df["passengers"] = df["passengers"].apply(parse_number)

    for c in ["fare", "fare_lg", "fare_low"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_money)

    for c in ["large_ms", "lf_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df[["city1_name", "city1_state"]] = df["city1"].apply(lambda x: pd.Series(extract_city_state(x)))
    df[["city2_name", "city2_state"]] = df["city2"].apply(lambda x: pd.Series(extract_city_state(x)))

    # Keep strict: do not invent dates, only keep valid Year/quarter rows.
    df = df[df["Year"].notna() & df["quarter"].notna()]
    df = df[df["fare"].notna() & df["nsmiles"].notna()]

    # Optional dedupe on common grain if those columns exist
    grain = ["Year","quarter","citymarketid_1","citymarketid_2","carrier_lg","carrier_low"]
    existing = [c for c in grain if c in df.columns]
    if len(existing) >= 4:
        df = df.drop_duplicates(subset=existing)

    return df


# ---------- external: jet fuel (weekly) -> quarterly ----------
def load_fuel_quarterly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "observation_date" not in df.columns:
        raise ValueError("Fuel file missing observation_date")
    val_col = [c for c in df.columns if c != "observation_date"][0]
    df = add_year_quarter(df, "observation_date")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    out = (df.groupby(["Year","quarter"], dropna=False)[val_col].mean()
             .reset_index()
             .rename(columns={val_col:"jet_fuel_price_gulf"}))
    out = out[out["Year"].notna() & out["quarter"].notna()]
    out["Year"] = out["Year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    return out


# ---------- external: labour/cpi-like (date -> quarterly) ----------
def load_cpi_quarterly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "observation_date" not in df.columns:
        raise ValueError("CPI file missing observation_date")
    val_col = [c for c in df.columns if c != "observation_date"][0]
    df = add_year_quarter(df, "observation_date")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    out = (df.groupby(["Year","quarter"], dropna=False)[val_col].mean()
             .reset_index()
             .rename(columns={val_col:"cpi_index"}))
    out = out[out["Year"].notna() & out["quarter"].notna()]
    out["Year"] = out["Year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    return out


# ---------- external: tourism (state-level) ----------
def load_overseas_visitors(path: Path) -> pd.DataFrame:
    """
    Parses '2024-Top-States-and-Cities-Visited.csv' (ranked table by state).
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    data_lines = [ln for ln in lines if re.match(r"^\s*\d+\s*,", ln)]
    if not data_lines:
        raise ValueError("Could not find ranked state rows in overseas visitors file.")

    rows = []
    for ln in data_lines:
        row = next(csv.reader([ln]))
        if len(row) < 7:
            continue

        def pct(p):
            p = str(p).strip().replace("%","")
            try: return float(p)/100.0
            except: return np.nan

        def num_k(n):
            n = str(n).strip().replace(",","").replace('"',"")
            if n == "": return np.nan
            try: return int(float(n)) * 1000  # file is in (000)
            except: return np.nan

        state_name = re.sub(r"^\s+", "", row[1].strip())
        abbr = _STATE_TO_ABBR.get(state_name)
        rows.append({
            "state_name": state_name,
            "state_abbr": abbr,
            "overseas_share_2024": pct(row[2]),
            "overseas_visitation_2024": num_k(row[3]),
            "overseas_change_2024_vs_2023": pct(row[4]),
            "overseas_share_2023": pct(row[5]),
            "overseas_visitation_2023": num_k(row[6]),
        })

    out = pd.DataFrame(rows)
    out = out[out["state_abbr"].notna()].drop_duplicates(subset=["state_abbr"])
    return out


# ---------- compile ----------
def compile_dataset(
    tickets_path: Path,
    fuel_path: Optional[Path],
    cpi_path: Optional[Path],
    tourism_path: Optional[Path],
) -> pd.DataFrame:

    tickets = load_clean_tickets(tickets_path)

    # Quarterly merges (LEFT JOIN keeps original rows)
    if fuel_path and fuel_path.exists():
        tickets = tickets.merge(load_fuel_quarterly(fuel_path), on=["Year","quarter"], how="left")

    if cpi_path and cpi_path.exists():
        tickets = tickets.merge(load_cpi_quarterly(cpi_path), on=["Year","quarter"], how="left")

    if tourism_path and tourism_path.exists():
        ov = load_overseas_visitors(tourism_path)
        tickets = tickets.merge(
            ov.add_prefix("orig_").rename(columns={"orig_state_abbr":"city1_state"}),
            on="city1_state",
            how="left"
        )
        tickets = tickets.merge(
            ov.add_prefix("dest_").rename(columns={"dest_state_abbr":"city2_state"}),
            on="city2_state",
            how="left"
        )

    # Modeling convenience features
    tickets["fare_per_mile"] = tickets["fare"] / tickets["nsmiles"]

    if "large_ms" in tickets.columns:
        tickets["dominance_bucket"] = pd.cut(
            tickets["large_ms"],
            bins=[-np.inf, 0.4, 0.7, np.inf],
            labels=["high_competition","moderate","dominated"]
        )

    if "lf_ms" in tickets.columns:
        tickets["lcc_bucket"] = pd.cut(
            tickets["lf_ms"],
            bins=[-np.inf, 0.15, 0.35, np.inf],
            labels=["low_lcc","medium_lcc","high_lcc"]
        )

    # Stable ordering
    tickets = tickets.sort_values(["Year","quarter","citymarketid_1","citymarketid_2"]).reset_index(drop=True)
    return tickets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=None,
                    help="Path to repo root. If omitted, inferred from this script location.")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output folder. Default: Data Cleaning/Processed")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else script_dir.parent

    # Inputs matching your folder tree
    tickets_path = repo_root / "Raw Data" / "Airline" / "airline_ticket_dataset.csv"
    fuel_path    = repo_root / "Raw Data" / "JetFuel" / "WJFUELUSGULF.csv"
    cpi_path     = repo_root / "Raw Data" / "LabourCost" / "CIS2024300000000I.csv"
    tourism_path = repo_root / "Raw Data" / "Tourism" / "2024-Top-States-and-Cities-Visited.csv"

    outdir = Path(args.outdir).resolve() if args.outdir else (repo_root / "Data Cleaning" / "Processed")
    outdir.mkdir(parents=True, exist_ok=True)

    df_final = compile_dataset(
        tickets_path=tickets_path,
        fuel_path=fuel_path,
        cpi_path=cpi_path,
        tourism_path=tourism_path,
    )

    # Write CSV always; Parquet only if engine exists
    csv_path = outdir / "final_dataset.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}  Rows={len(df_final):,} Cols={df_final.shape[1]}")

    parquet_path = outdir / "final_dataset.parquet"
    try:
        df_final.to_parquet(parquet_path, index=False)
        print(f"Wrote: {parquet_path}")
    except Exception as e:
        print("Parquet skipped (install pyarrow/fastparquet to enable):", e)


if __name__ == "__main__":
    main()