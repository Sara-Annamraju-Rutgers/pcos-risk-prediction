import pandas as pd, pathlib, textwrap

# ❶ Load the lifestyle file
csv = pathlib.Path("data") / "lifestyle-pcos.csv"
df  = pd.read_csv(csv)

# ❷ Identify the target column
target = "have_you_been_diagnosed_with_pcos_pcod"   # the yes/no column

# ❸ Compute balance
pos_pct = 100 * (df[target] == 1).mean()
print(textwrap.dedent(f"""
    Positive‑class share (diagnosed = 1):
        {pos_pct:.1f} %
"""))