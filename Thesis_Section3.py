import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. Data Loading and Cleaning
# ==========================================
df = pd.read_csv('vote_TK25.csv', sep=';')
df_parties = df[df['Type'] == 'Partij'].copy()
df_parties = df_parties[['Partij', 'AantalStemmen']].reset_index(drop=True)

votes = df_parties['AantalStemmen'].values
parties = df_parties['Partij'].values
V_total = votes.sum()
vote_shares = (votes / V_total) * 100
total_seats = 150


# ==========================================
# 2. Core Allocation Algorithm
# ==========================================
def dhondt(votes, total_seats=150):
    """D'Hondt method for seat allocation."""
    votes = np.array(votes, dtype=float)
    seats = np.zeros(len(votes), dtype=int)
    if np.sum(votes) == 0: return seats

    for _ in range(total_seats):
        divisors = seats + 1.0
        quotients = np.where(votes > 0, votes / divisors, -1)
        winner_idx = np.argmax(quotients)
        seats[winner_idx] += 1
    return seats


# ==========================================
# 3. Evaluation Functions
# ==========================================
v_props = vote_shares / 100.0
ENPV_constant = 1.0 / np.sum(v_props ** 2)


def simulate_election(threshold):
    """Calculates metrics based on electoral threshold."""
    eligible_mask = vote_shares >= threshold
    filtered_votes = np.where(eligible_mask, votes, 0)
    seats = dhondt(filtered_votes, total_seats=total_seats)

    seat_shares = (seats / total_seats) * 100
    s_props = seat_shares / 100.0

    # Effective Number of Parliamentary Parties
    enpp = 1.0 / np.sum(s_props[s_props > 0] ** 2) if np.sum(s_props) > 0 else 0
    # Gallagher Index (Least Squares Index)
    lsq = np.sqrt(0.5 * np.sum((vote_shares - seat_shares) ** 2))
    # Total parliamentary parties
    n_parties = np.count_nonzero(seats)

    return ENPV_constant, enpp, lsq, n_parties


# ==========================================
# 4. Iteration and Smoothing
# ==========================================
thresholds = np.linspace(0.67, 5.0, 1000)
enpv_results, enpp_results, lsq_results, n_parties_results = [], [], [], []

for t in thresholds:
    enpv, enpp, lsq, n_parties = simulate_election(t)
    enpv_results.append(enpv)
    enpp_results.append(enpp)
    lsq_results.append(lsq)
    n_parties_results.append(n_parties)

# Apply Gaussian filter for cleaner visualization
sigma_val = 15
enpp_smooth = gaussian_filter1d(enpp_results, sigma=sigma_val)
lsq_smooth = gaussian_filter1d(lsq_results, sigma=sigma_val)
n_parties_smooth = gaussian_filter1d(n_parties_results, sigma=sigma_val)

# ==========================================
# 5. Visualization
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Electoral Threshold (%)', fontsize=12)
ax1.set_ylabel('Number of Parties', color='black', fontsize=12)

# Plot ENPV, ENPP, and total parliamentary parties
ax1.plot(thresholds, enpv_results, color='tab:green', linewidth=2, label='ENPV')
ax1.plot(thresholds, enpp_smooth, color='tab:blue', linewidth=2, label='ENPP')
ax1.plot(thresholds, n_parties_smooth, color='tab:purple', linewidth=2, label='Total Parliamentary Parties')
ax1.tick_params(axis='y', labelcolor='black')

# Secondary axis for Gallagher Index
ax2 = ax1.twinx()
ax2.set_ylabel('Gallagher Index ($LSq$)', color='black', fontsize=12)
ax2.plot(thresholds, lsq_smooth, color='tab:red', linewidth=2, label='$LSq$')
ax2.tick_params(axis='y', labelcolor='black')

fig.suptitle('Sensitivity Analysis of Electoral Thresholds', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.grid(True, alpha=0.3)

# Merge legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=12)

plt.show()