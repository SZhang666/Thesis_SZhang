import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.nonparametric.smoothers_lowess import lowess

# ============================================================
# 1. Data Loading
# ============================================================

file_path = "vote_share.csv"
raw = pd.read_csv(file_path, header=None)

headers = raw.iloc[0].tolist()
alliances = raw.iloc[2].tolist()
data = raw.iloc[3:].copy()
data.columns = headers

# ============================================================
# 2. Setup Columns and Alliance Mapping
# ============================================================

meta_cols = ["Polling firm", "Fieldwork date", "Sample\nsize", "Lead"]
party_cols = [c for c in data.columns if c not in meta_cols]

party_to_alliance = {
    col: (alliance if col != "Others" else "Others")
    for col, alliance in zip(headers, alliances) if col in party_cols
}


# ============================================================
# 3. Clean Vote-Share Values
# ============================================================

def percent_to_float(x):
    """Clean percentage strings."""
    if pd.isna(x): return np.nan
    x = str(x).replace("%", "").replace("–", "").replace("-", "").strip()
    return float(x) if x != "" else np.nan


for col in party_cols:
    data[col] = data[col].apply(percent_to_float)


# ============================================================
# 4. Parse Fieldwork Dates
# ============================================================

def parse_fieldwork_end_date(s):
    """Extract end date from ranges."""
    if pd.isna(s): return pd.NaT
    s = str(s).replace("[*]", "").replace("–", "-").strip()
    if "-" in s:
        right = s.split("-")[-1].strip()
        return pd.to_datetime(right, errors="coerce")
    return pd.to_datetime(s, format="%d %b %Y", errors="coerce")


data["date"] = data["Fieldwork date"].apply(parse_fieldwork_end_date)
data_no_exit = data[~data["Fieldwork date"].astype(str).str.contains(r"\[\*\]", na=False)].copy()

# ============================================================
# 5. Aggregate by Bloc
# ============================================================

bloc_names = list(dict.fromkeys(party_to_alliance.values()))
bloc_df = data_no_exit[["date"]].copy()

for bloc in bloc_names:
    cols = [p for p, b in party_to_alliance.items() if b == bloc]
    bloc_df[bloc] = data_no_exit[cols].sum(axis=1, skipna=True)

bloc_df = bloc_df.sort_values("date")

# ============================================================
# 6. Visualization with LOWESS Smoothing
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7), layout='constrained')
smooth_frac = 0.25

for bloc in bloc_names:
    plot_data = bloc_df[["date", bloc]].dropna().copy()
    x_dates = plot_data["date"]
    x_numeric = mdates.date2num(x_dates)
    y = plot_data[bloc].values

    ax.scatter(x_dates, y, s=35, alpha=0.35)
    smoothed = lowess(y, x_numeric, frac=smooth_frac, return_sorted=True)
    ax.plot(mdates.num2date(smoothed[:, 0]), smoothed[:, 1], label=bloc, linewidth=2.2)

ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Vote share (%)", fontsize=14)
ax.set_title("Polling Vote Share per Bloc (TK23 to TK25)", fontsize=14, fontweight='bold')
ax.legend(title="Bloc", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=18, frameon=True)
ax.grid(True, alpha=0.3)

plt.show()

# ============================================================
# 7. Statistical Summary by Bloc
# ============================================================

party_sds = {party: data_no_exit[party].std() for party in party_cols}
results = []

for bloc in bloc_names:
    parties = [p for p, b in party_to_alliance.items() if b == bloc]
    sds = [party_sds[p] for p in parties if pd.notna(party_sds[p])]

    if sds:
        max_sd = np.max(sds)
        max_p = [p for p in parties if party_sds[p] == max_sd][0]
        results.append({
            "Bloc": bloc, "N": len(sds),
            "Average SD": np.mean(sds), "Median SD": np.median(sds),
            "Max SD": f"{max_sd:.3f}({max_p})"
        })

results_df = pd.DataFrame(results).sort_values(by="Average SD", ascending=False)
print("\n--- Standard Deviation Summary by Bloc ---")
print(results_df.to_string(index=False))