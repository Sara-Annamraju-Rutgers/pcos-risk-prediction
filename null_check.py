import pandas as pd, pathlib

csv_path = pathlib.Path("data") / "lifestyle-pcos.csv"
df = pd.read_csv(csv_path)

null_pct = df.isna().mean().mul(100).round(1)
missing = null_pct[null_pct > 0].sort_values(ascending=False)

print("\nMissing‑value report:")
print(" None missing!" if missing.empty else missing)
