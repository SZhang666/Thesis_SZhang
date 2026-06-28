import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Seat Allocation Core Algorithms
# ==========================================

def dhondt(votes, total_seats, quota):
    # Initial allocation based on quota
    initial_seats = [int(v // quota) for v in votes]
    remainder_seats = [0] * len(votes)
    current_seats = initial_seats.copy()
    remaining_seats = total_seats - sum(initial_seats)

    # Distribute remaining seats using highest average method
    for _ in range(remaining_seats):
        quotients = [v / (s + 1) for v, s in zip(votes, current_seats)]
        winner_idx = quotients.index(max(quotients))
        remainder_seats[winner_idx] += 1
        current_seats[winner_idx] += 1
    return initial_seats, remainder_seats


def sainte_lague(votes, total_seats, quota):
    initial_seats = [int(v // quota) for v in votes]
    remainder_seats = [0] * len(votes)
    current_seats = initial_seats.copy()
    remaining_seats = total_seats - sum(initial_seats)

    for _ in range(remaining_seats):
        quotients = [v / (2 * s + 1) for v, s in zip(votes, current_seats)]
        winner_idx = quotients.index(max(quotients))
        remainder_seats[winner_idx] += 1
        current_seats[winner_idx] += 1
    return initial_seats, remainder_seats


def modified_sainte_lague(votes, total_seats, quota):
    initial_seats = [int(v // quota) for v in votes]
    remainder_seats = [0] * len(votes)
    current_seats = initial_seats.copy()
    remaining_seats = total_seats - sum(initial_seats)

    for _ in range(remaining_seats):
        quotients = []
        for v, s in zip(votes, current_seats):
            if s == 0:
                quotients.append(v / 1.4)
            else:
                quotients.append(v / (2 * s + 1))
        winner_idx = quotients.index(max(quotients))
        remainder_seats[winner_idx] += 1
        current_seats[winner_idx] += 1
    return initial_seats, remainder_seats


def hare_quota(votes, total_seats):
    initial_seats = [0] * len(votes)
    remainder_seats = [0] * len(votes)
    total_votes = sum(votes)
    quota = total_votes / total_seats

    # Calculate initial seats and remainders for largest remainder method
    remainders = []
    for i, v in enumerate(votes):
        initial_seats[i] = int(v // quota)
        remainders.append([v % quota, i])

    remaining_seats = total_seats - sum(initial_seats)
    remainders.sort(key=lambda x: x[0], reverse=True)

    for i in range(remaining_seats):
        winner_idx = remainders[i][1]
        remainder_seats[winner_idx] += 1
    return initial_seats, remainder_seats


def droop_quota(votes, total_seats):
    initial_seats = [0] * len(votes)
    remainder_seats = [0] * len(votes)
    total_votes = sum(votes)
    quota = (total_votes // (total_seats + 1)) + 1

    remainders = []
    for i, v in enumerate(votes):
        initial_seats[i] = int(v // quota)
        remainders.append([v % quota, i])

    remaining_seats = total_seats - sum(initial_seats)
    if remaining_seats > 0:
        remainders.sort(key=lambda x: x[0], reverse=True)
        for i in range(remaining_seats):
            winner_idx = remainders[i][1]
            remainder_seats[winner_idx] += 1
    return initial_seats, remainder_seats


def imperiali_quota(votes, total_seats):
    initial_seats = [0] * len(votes)
    remainder_seats = [0] * len(votes)
    total_votes = sum(votes)
    quota = total_votes / (total_seats + 2)

    remainders = []
    for i, v in enumerate(votes):
        initial_seats[i] = int(v // quota)
        remainders.append([v % quota, i])

    remaining_seats = total_seats - sum(initial_seats)
    if remaining_seats > 0:
        remainders.sort(key=lambda x: x[0], reverse=True)
        for i in range(remaining_seats):
            winner_idx = remainders[i][1]
            remainder_seats[winner_idx] += 1
    return initial_seats, remainder_seats


# ==========================================
# 2. Data Preparation
# ==========================================
data = {
    'Party': [
        'D66', 'PVV', 'VVD', 'GL/PvdA', 'CDA', 'JA21',
        'FvD', 'BBB', 'DENK', 'SGP', 'PvdD', 'CU', 'SP',
        '50PLUS', 'Volt', 'BIJ1', 'NSC', 'BVNL',
        'Vrede voor Dieren', 'Piratenpartij', 'FNP',
        'LP', 'DE LINIE', 'NL PLAN', 'Vrij Verbond',
        'ELLECT', 'Partij voor de Rechtsstaat'
    ],
    'Votes': [
        1790634, 1760966, 1505829, 1352163, 1246874, 628517,
        480393, 279916, 250368, 238093, 219371, 201361, 199585,
        151053, 116468, 40360, 39408, 18477,
        16819, 10575, 9331,
        8248, 3478, 2299, 1048,
        205, 151
    ]
}
df = pd.DataFrame(data)
total_seats = 150
total_valid_votes = df['Votes'].sum()
kiesdeler = total_valid_votes / total_seats

# Filter parties meeting the natural threshold
df_eligible = df[df['Votes'] >= kiesdeler].copy()
eligible_votes = df_eligible['Votes'].values
parties = df_eligible['Party'].values

# ==========================================
# 3. Visualization
# ==========================================
methods = [
    ("(a) D'Hondt", dhondt, True),
    ("(b) Sainte-Laguë", sainte_lague, True),
    ("(c) (Modified) Sainte-Laguë", modified_sainte_lague, True),
    ("(d) Hare Quota", hare_quota, False),
    ("(e) Droop Quota", droop_quota, False),
    ("(f) Imperiali Quota", imperiali_quota, False)
]

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axes = axes.flatten()

for i, (method_name, func, needs_quota_param) in enumerate(methods):
    ax = axes[i]

    if needs_quota_param:
        initial, remainder = func(eligible_votes, total_seats, kiesdeler)
    else:
        initial, remainder = func(eligible_votes, total_seats)

    initial = np.array(initial)
    remainder = np.array(remainder)
    total_allocated = initial + remainder

    # Filter parties with at least one seat
    valid_indices = total_allocated > 0
    plot_parties = parties[valid_indices]
    plot_initial = initial[valid_indices]
    plot_remainder = remainder[valid_indices]
    plot_total = total_allocated[valid_indices]

    # Create stacked bar chart
    bars1 = ax.bar(plot_parties, plot_initial, label='Initial Seats', color='#64766A', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(plot_parties, plot_remainder, bottom=plot_initial, label='Slack Seats', color='#C0A9BD',
                   edgecolor='white', linewidth=0.5)

    # Label individual seat counts
    for j in range(len(plot_parties)):
        if plot_initial[j] > 0:
            ax.text(j, plot_initial[j] / 2, str(plot_initial[j]),
                    ha='center', va='center', color='white', fontsize=12, fontweight='bold')

        if plot_remainder[j] > 0:
            y_pos = plot_initial[j] + plot_remainder[j] / 2
            ax.text(j, y_pos, str(plot_remainder[j]),
                    ha='center', va='center', color='black', fontsize=12, fontweight='bold')

    ax.set_title(method_name, fontsize=16, fontweight='bold')
    ax.set_ylabel('Seats')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(plot_total) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if i == 0:
        ax.legend(loc='upper right', fontsize=16)

plt.tight_layout()
plt.show()