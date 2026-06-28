import numpy as np
import pandas as pd


# =========================================================
# 1. Seat Allocation & Formatting Utilities
# =========================================================

def dhondt(votes, total_seats, quota):
    """Calculate seat distribution using D'Hondt method."""
    initial_seats = [int(v // quota) for v in votes]
    remainder_seats = [0] * len(votes)
    current_seats = initial_seats.copy()
    remaining_seats = total_seats - sum(initial_seats)

    for _ in range(remaining_seats):
        quotients = [v / (s + 1) for v, s in zip(votes, current_seats)]
        winner_idx = quotients.index(max(quotients))
        remainder_seats[winner_idx] += 1
        current_seats[winner_idx] += 1

    return initial_seats, remainder_seats


def classify_seat_source(party_index, total_diff, initial_diff, remainder_diff):
    """Determine if seat change stems from initial allocation or remainder (slack)."""
    td = total_diff[party_index]
    idiff = initial_diff[party_index]
    rdiff = remainder_diff[party_index]

    if td == 0: return None
    initial_same_direction = (idiff > 0 and td > 0) or (idiff < 0 and td < 0)
    remainder_same_direction = (rdiff > 0 and td > 0) or (rdiff < 0 and td < 0)

    if initial_same_direction and not remainder_same_direction:
        return "initial"
    elif remainder_same_direction and not initial_same_direction:
        return "slack"
    elif initial_same_direction and remainder_same_direction:
        return "initial + slack"
    else:
        return "mixed"


def format_losing_gaining(parties, total_diff, initial_diff, remainder_diff):
    """Format descriptions for seat gains and losses."""
    losing_items, gaining_items = [], []
    for i, d in enumerate(total_diff):
        if d == 0: continue
        source = classify_seat_source(i, total_diff, initial_diff, remainder_diff)
        if d < 0:
            losing_items.append(f"{parties[i]} ↓{abs(int(d))} ({source})")
        elif d > 0:
            gaining_items.append(f"{parties[i]} ↑{int(d)} ({source})")

    return ", ".join(losing_items) if losing_items else "None", ", ".join(gaining_items) if gaining_items else "None"


def format_final_transfers(parties, total_diff):
    """Format flow of seats between parties."""
    losing, gaining = [], []
    for i, d in enumerate(total_diff):
        if d < 0:
            losing.append([parties[i], abs(int(d))])
        elif d > 0:
            gaining.append([parties[i], int(d)])

    if not losing and not gaining: return "None"

    transfers, l, g = [], 0, 0
    while l < len(losing) and g < len(gaining):
        loser, lost_count = losing[l]
        winner, gained_count = gaining[g]
        move_count = min(lost_count, gained_count)
        transfers.append(f"{loser} → {winner} ({move_count})")
        losing[l][1] -= move_count
        gaining[g][1] -= move_count
        if losing[l][1] == 0: l += 1
        if gaining[g][1] == 0: g += 1
    return "; ".join(transfers)


def format_internal_swaps(parties, total_diff, initial_diff, remainder_diff):
    """Format changes where total seats remain constant but allocation structure shifts."""
    swap_items = []
    for i in range(len(parties)):
        if total_diff[i] == 0 and (initial_diff[i] != 0 or remainder_diff[i] != 0):
            init_change = int(initial_diff[i])
            slack_change = int(remainder_diff[i])
            init_text = f"init ↑{init_change}" if init_change > 0 else f"init ↓{abs(init_change)}"
            slack_text = f"slack ↑{slack_change}" if slack_change > 0 else f"slack ↓{abs(slack_change)}"
            swap_items.append(f"{parties[i]}: {init_text}, {slack_text}")
    return ", ".join(swap_items) if swap_items else "None"


# =========================================================
# 2. Sensitivity Analysis Core
# =========================================================

def run_sub3_sensitivity_analysis():
    # Load data
    df = pd.read_csv('vote_paper.csv', sep=';')

    y_B = float(df[df['Type'] == 'AantalBlancoStemmen']['AantalStemmen'].iloc[0])
    y_INV = float(df[df['Type'] == 'AantalOngeldigeStemmen']['AantalStemmen'].iloc[0])

    parties_df = df[df['Type'] == 'Partij'].copy().reset_index(drop=True)
    party_names_full = parties_df['Partij'].tolist()
    parties = [p.split('(')[0].strip() if '(' in p else p for p in party_names_full]
    Y_P = parties_df['AantalStemmen'].values.astype(float)

    official_total_seats = pd.to_numeric(parties_df['AantalZetels'], errors='coerce').fillna(0).astype(int).values

    n = len(Y_P)
    S_y = np.sum(Y_P)
    total_seats = 150
    total = 10571990

    # Calculate baseline seats
    reference_quota = S_y / total_seats
    ref_initial, ref_remainder = dhondt(Y_P, total_seats, reference_quota)
    ref_initial = np.array(ref_initial, dtype=int)
    ref_remainder = np.array(ref_remainder, dtype=int)

    # Fixed model parameters
    zeta = 0.0005
    zeta_P, zeta_INV = 0.8 * zeta, 0.2 * zeta
    eta = 0.005
    eta_P, eta_B = 0.8 * eta, 0.2 * eta

    p = np.full(n, 0.5)
    p[0], p[-1] = 0.0, 1.0

    rows = []

    # Run loop
    for error_votes in range(0, 200001, 500):
        # 1. Update epsilon distributions
        eps = error_votes / total
        eps_P = 0.95 * eps
        eps_B = 0.2 * (eps - eps_P)
        eps_INV = 0.8 * (eps - eps_P)

        # 2. Macro adjustment
        A_macro = np.array([[1 - eps_B - eps_INV, zeta_P, eta_P],
                            [eps_B, 1 - zeta, eta_B],
                            [eps_INV, zeta_INV, 1 - eta]])
        Y_macro = np.array([S_y, y_B, y_INV])

        try:
            X_macro_hat = np.linalg.inv(A_macro) @ Y_macro
            S_x_hat, x_B_hat, x_INV_hat = X_macro_hat
        except np.linalg.LinAlgError:
            continue

        # 3. Party-specific adjustment matrix
        C = (zeta_P * x_B_hat + eta_P * x_INV_hat) / S_x_hat
        M_hat = np.zeros((n, n))
        for j in range(n):
            M_hat[j, j] = 1 - eps + C
            if j > 0: M_hat[j - 1, j] = eps_P * p[j]
            if j < n - 1: M_hat[j + 1, j] = eps_P * (1 - p[j])

        try:
            X_P_hat = np.maximum(np.linalg.inv(M_hat) @ Y_P, 0)
        except np.linalg.LinAlgError:
            continue

        # 4. Seat redistribution and change analysis
        quota_eps = np.sum(X_P_hat) / total_seats
        initial_eps, remainder_eps = dhondt(X_P_hat, total_seats, quota_eps)
        total_diff = (np.array(initial_eps) + np.array(remainder_eps)) - official_total_seats

        rows.append({
            "Error Votes": int(error_votes),
            "Epsilon (%)": eps * 100,
            "Total Seat Changes": int(np.sum(np.abs(total_diff)) / 2),
            "Losing Seats": format_losing_gaining(parties, total_diff, np.array(initial_eps) - ref_initial,
                                                  np.array(remainder_eps) - ref_remainder)[0],
            "Gaining Seats": format_losing_gaining(parties, total_diff, np.array(initial_eps) - ref_initial,
                                                   np.array(remainder_eps) - ref_remainder)[1],
            "Final Transfers": format_final_transfers(parties, total_diff),
            "Internal Swaps": format_internal_swaps(parties, total_diff, np.array(initial_eps) - ref_initial,
                                                    np.array(remainder_eps) - ref_remainder)
        })

    # Output results
    summary_df = pd.DataFrame(rows)
    print("=" * 140)
    print("SUB3 SENSITIVITY ANALYSIS")
    print("=" * 140)
    print(summary_df.to_string(index=False,
                               formatters={"Error Votes": lambda x: f"{x:,}", "Epsilon (%)": lambda x: f"{x:.4f}%"}))
    print("=" * 140)


if __name__ == "__main__":
    run_sub3_sensitivity_analysis()