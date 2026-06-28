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
data = raw.iloc[3:].copy()
data.columns = headers

# ============================================================
# 2. Setup Columns
# ============================================================

meta_cols = ["Polling firm", "Fieldwork date", "Sample\nsize", "Lead"]
party_cols = [c for c in data.columns if c not in meta_cols and c != "Others"]


# ============================================================
# 3. Clean Vote-Share Values
# ============================================================

def percent_to_float(x):
    """Convert percentage string to float."""
    if pd.isna(x): return np.nan
    x = str(x).replace("%", "").replace("–", "").replace("-", "").strip()
    return float(x) if x != "" else np.nan


for col in party_cols:
    data[col] = data[col].apply(percent_to_float)


# ============================================================
# 4. Parse Fieldwork Dates
# ============================================================

def parse_fieldwork_end_date(s):
    """Parse date ranges to end date."""
    if pd.isna(s): return pd.NaT
    s = str(s).replace("[*]", "").replace("–", "-").strip()

    if "-" in s:
        right = s.split("-")[-1].strip()
        return pd.to_datetime(right, errors="coerce")
    return pd.to_datetime(s, format="%d %b %Y", errors="coerce")


data["date"] = data["Fieldwork date"].apply(parse_fieldwork_end_date)

# ============================================================
# 5. Preprocessing
# ============================================================

# Remove exit poll entries and sort by date
data_no_exit = data[~data["Fieldwork date"].astype(str).str.contains(r"\[\*\]", na=False)].copy()
data_no_exit = data_no_exit.sort_values("date")

# ============================================================
# 6. Visualization with LOWESS Smoothing
# ============================================================

fig, ax = plt.subplots(figsize=(16, 9), layout='constrained')
smooth_frac = 0.25

for party in party_cols:
    plot_data = data_no_exit[["date", party]].dropna().copy()
    x_dates = plot_data["date"]
    x_numeric = mdates.date2num(x_dates)
    y = plot_data[party].values

    # Plot raw data points and trend curve
    ax.scatter(x_dates, y, s=22, alpha=0.28)
    smoothed = lowess(y, x_numeric, frac=smooth_frac, return_sorted=True)

    ax.plot(mdates.num2date(smoothed[:, 0]), smoothed[:, 1], label=party, linewidth=1.8)

# Styling
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Vote share (%)", fontsize=16)
ax.set_title("Polling Vote Share per Party (TK23 to TK25)", fontsize=16, fontweight='bold')

# Axis formatting
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Legend and Grid
ax.legend(title="Party", loc="center left", bbox_to_anchor=(1.01, 0.5),
          ncol=2, fontsize=18, title_fontsize=18, frameon=True)
ax.grid(True, alpha=0.3)

plt.show()