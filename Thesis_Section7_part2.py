import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Load and Prepare Data
# ============================================================

raw = pd.read_csv("vote_share.csv", header=None)

headers = raw.iloc[0].tolist()
alliances = raw.iloc[2].tolist()
data = raw.iloc[3:].copy()
data.columns = headers

meta_cols = ["Polling firm", "Fieldwork date", "Sample\nsize", "Lead"]
party_cols_all = [c for c in data.columns if c not in meta_cols]

# Map parties to blocs
party_to_bloc = {col: bloc for col, bloc in zip(headers, alliances)
                 if col in party_cols_all and col != "Others"}
party_cols = list(party_to_bloc.keys())

# ============================================================
# 2. Clean and Filter Data
# ============================================================

def percent_to_float(x):
    """Clean percentage string and convert to float."""
    if pd.isna(x): return np.nan
    x = str(x).replace("%", "").replace("–", "").replace("-", "").strip()
    return float(x) if x != "" else np.nan

for col in party_cols:
    data[col] = data[col].apply(percent_to_float)

# Filter out exit polls
analysis_df = data[~data["Fieldwork date"].astype(str).str.contains(r"\[\*\]", na=False)].copy()

# ============================================================
# 3. Statistical Analysis
# ============================================================

party_stats = []
for party in party_cols:
    series = analysis_df[party].dropna()
    party_stats.append({
        "Party": party,
        "Bloc": party_to_bloc[party],
        "Mean": series.mean(),
        "SD": series.std(ddof=1),
        "Range": series.max() - series.min(),
        "Min": series.min(),
        "Max": series.max(),
        "N": int(series.count())
    })

party_volatility = pd.DataFrame(party_stats).sort_values("SD", ascending=False).reset_index(drop=True)

# Print Summary Tables
print("\n--- Party-level Volatility ---")
print(party_volatility.round(3).to_string(index=False))

# ============================================================
# 4. Visualization
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8), layout='constrained')
x = np.arange(len(party_volatility))

# Plot SD bars and Min-Max ranges
ax.bar(x, party_volatility["SD"], color=plt.cm.YlGnBu(np.linspace(0.45, 0.85, len(party_volatility))),
       width=0.62, label="Vote-share SD", zorder=3)
ax.fill_between(x, party_volatility["Min"], party_volatility["Max"],
                color="steelblue", alpha=0.12, label="Min–Max interval", zorder=1)

# Plot Min/Max lines
ax.plot(x, party_volatility["Max"], color="#1f77b4", marker="o", linewidth=2.0, label="Max vote share", zorder=4)
ax.plot(x, party_volatility["Min"], color="#8ecae6", linestyle="--", marker="o", linewidth=2.0, label="Min vote share", zorder=4)

# Annotate bars
for i, row in party_volatility.iterrows():
    ax.text(x[i], row["SD"] + 0.1, f"{row['SD']:.2f}", ha="center", fontsize=14, fontweight="bold")

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(party_volatility["Party"], rotation=45, ha="right", fontsize=18)
ax.set_xlabel("Party", fontsize=18)
ax.set_ylabel("Vote share (%)", fontsize=18)
ax.set_title("Polling Vote Share Volatility (TK23 to TK25)", fontsize=18, fontweight='bold', pad=15)

# Legends and Text
ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.98), fontsize=20, framealpha=0.9)
ax.text(0.99, 0.72, "Bars = SD\nSolid line = Max\nDashed line = Min\nShaded = Range",
        transform=ax.transAxes, ha="right", va="top", fontsize=20,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.9, facecolor='white'))

ax.grid(axis="y", alpha=0.2)
plt.show()