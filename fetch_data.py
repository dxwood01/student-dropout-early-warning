"""Run once: downloads the UCI dataset (id 697) into data/students_raw.csv."""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

d = fetch_ucirepo(id=697)
df = d.data.features.copy()
df["Target"] = d.data.targets["Target"].values
out = Path(__file__).parent / "data" / "students_raw.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"saved {out}  ({df.shape[0]} rows, {df.shape[1]} columns)")
