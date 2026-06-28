import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# ==========================================
# 1. Core Allocation Algorithms
# ==========================================

def dhondt(votes, total_seats=150):
    """Highest averages method (D'Hondt)."""
    seats = np.zeros(len(votes), dtype=int)
    for _ in range(total_seats):
        quotients = np.where(votes > 0, votes / (seats + 1.0), -1)
        seats[np.argmax(quotients)] += 1
    return seats


def sainte_lague(votes, total_seats=150):
    """Highest averages method (Sainte-Laguë)."""
    seats = np.zeros(len(votes), dtype=int)
    for _ in range(total_seats):
        quotients = np.where(votes > 0, votes / (2.0 * seats + 1.0), -1)
        seats[np.argmax(quotients)] += 1
    return seats


def modified_sainte_lague(votes, total_seats=150):
    """Modified Sainte-Laguë (divisor 1.4 for first seat)."""
    seats = np.zeros(len(votes), dtype=int)
    for _ in range(total_seats):
        divisors = np.where(seats == 0, 1.4, 2.0 * seats + 1.0)
        quotients = np.where(votes > 0, votes / divisors, -1)
        seats[np.argmax(quotients)] += 1
    return seats


def hare_quota(votes, total_seats=150):
    """Largest Remainder Method (Hare Quota)."""
    seats = np.zeros(len(votes), dtype=int)
    total_eligible = np.sum(votes)
    if total_eligible == 0: return seats
    quota = total_eligible / total_seats

    seats += np.floor(votes / quota).astype(int)
    remainders = np.where(votes > 0, votes % quota, -1)

    for _ in range(total_seats - np.sum(seats)):
        winner = np.argmax(remainders)
        seats[winner] += 1
        remainders[winner] = -1
    return seats


def droop_quota(votes, total_seats=150):
    """Largest Remainder Method (Droop Quota)."""
    seats = np.zeros(len(votes), dtype=int)
    total_eligible = np.sum(votes)
    if total_eligible == 0: return seats
    quota = np.floor(total_eligible / (total_seats + 1)) + 1

    seats += np.floor(votes / quota).astype(int)
    remainders = np.where(votes > 0, votes % quota, -1)

    for _ in range(total_seats - np.sum(seats)):
        winner = np.argmax(remainders)
        seats[winner] += 1
        remainders[winner] = -1
    return seats


def imperiali_quota(votes, total_seats=150):
    """Largest Remainder Method (Imperiali Quota)."""
    seats = np.zeros(len(votes), dtype=int)
    total_eligible = np.sum(votes)
    if total_eligible == 0: return seats
    quota = total_eligible / (total_seats + 2)

    seats += np.floor(votes / quota).astype(int)
    remainders = np.where(votes > 0, votes % quota, -1)

    for _ in range(total_seats - np.sum(seats)):
        if np.max(remainders) < 0: break
        winner = np.argmax(remainders)
        seats[winner] += 1
        remainders[winner] = -1
    return seats


# ==========================================
# 2. Data Preparation
# ==========================================
data = [
    ("D66", 1790634), ("PVV", 1760966), ("VVD", 1505829), ("GL/PvdA", 1352163),
    ("CDA", 1246874), ("JA21", 628517), ("FvD", 480393), ("BBB", 279916),
    ("DENK", 250368), ("SGP", 238093), ("PvdD", 219371), ("CU", 201361),
    ("SP", 199585), ("50PLUS", 151053), ("Volt", 116468), ("BIJ1", 40360),
    ("NSC", 39408), ("BVNL", 18477), ("Vrede voor Dieren", 16819), ("Piratenpartij", 10575),
    ("FNP", 9331), ("LP", 8248), ("DE LINIE", 3478), ("NL PLAN", 2299),
    ("Vrij Verbond", 1048), ("ELLECT", 205), ("Partij voor de Rechtsstaat", 151)
]

votes = np.array([x[1] for x in data], dtype=float)
vote_shares = (votes / votes.sum()) * 100
total_seats = 150
ENPV_constant = 1.0 / np.sum((vote_shares / 100.0) ** 2)

# ==========================================
# 3. Sensitivity Analysis Simulation
# ==========================================
methods_config = [
    ("D'Hondt", dhondt, '-'), ("Sainte-Laguë", sainte_lague, '-'),
    ("(M)Sainte-Laguë", modified_sainte_lague, '-'), ("Hare Quota", hare_quota, '--'),
    ("Droop Quota", droop_quota, '--'), ("Imperiali Quota", imperiali_quota, '--')
]

thresholds = np.linspace(0.67, 5.0, 500)
results = {m[0]: {'enpp': [], 'lsq': []} for m in methods_config}

for t in thresholds:
    eligible_votes = np.where(vote_shares >= t, votes, 0)
    for name, func, _ in methods_config:
        seats = func(eligible_votes, total_seats)
        s_props = (seats / total_seats)

        enpp = 1.0 / np.sum(s_props[s_props > 0] ** 2) if np.sum(s_props) > 0 else 0
        lsq = np.sqrt(0.5 * np.sum((vote_shares / 100.0 - s_props) ** 2))

        results[name]['enpp'].append(enpp)
        results[name]['lsq'].append(lsq)

# ==========================================
# 4. Visualization
# ==========================================
fig, ax1 = plt.subplots(figsize=(16, 9))
sigma_val = 8

# Left Axis: ENPP
ax1.set_xlabel('Electoral Threshold (%)', fontsize=14)
ax1.set_ylabel('Effective Number of Parties (ENPP)', fontsize=14)
ax1.plot(thresholds, [ENPV_constant] * len(thresholds), 'k:', linewidth=3, label='ENPV (Baseline)')

colors_enpp = ['#0000FF', '#008000', '#800080', '#00CED1', '#2F4F4F', '#3CB371']
for i, (name, _, linestyle) in enumerate(methods_config):
    smooth = gaussian_filter1d(results[name]['enpp'], sigma=sigma_val)
    ax1.plot(thresholds, smooth, color=colors_enpp[i], linestyle=linestyle, linewidth=2.5, label=f"{name} (ENPP)")

# Right Axis: Gallagher Index
ax2 = ax1.twinx()
ax2.set_ylabel('Gallagher Index (LSq)', fontsize=14)
colors_lsq = ['#FF0000', '#FF8C00', '#FF1493', '#8B4513', '#D4AF37', '#FF6347']

for i, (name, _, linestyle) in enumerate(methods_config):
    smooth = gaussian_filter1d(results[name]['lsq'], sigma=sigma_val)
    ax2.plot(thresholds, smooth, color=colors_lsq[i], linestyle=linestyle, linewidth=2.5, label=f"{name} (LSq)")

fig.suptitle('Sensitivity Analysis of Electoral Thresholds', fontsize=18, fontweight='bold')
ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()