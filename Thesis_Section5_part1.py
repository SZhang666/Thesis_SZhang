import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_voting_errors(csv_filepath='vote_paper.csv'):
    """
    Analyzes voting errors based on a deterministic counting error simulation.
    """
    # 1. Load and prepare data
    try:
        df = pd.read_csv(csv_filepath, sep=';')
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_filepath}'.")
        return

    try:
        y_B = float(df[df['Type'] == 'AantalBlancoStemmen']['AantalStemmen'].iloc[0])
        y_INV = float(df[df['Type'] == 'AantalOngeldigeStemmen']['AantalStemmen'].iloc[0])
    except IndexError:
        print("Error: Required vote types not found in dataset.")
        return

    parties_df = df[df['Type'] == 'Partij'].copy().reset_index(drop=True)
    party_names = parties_df['Partij'].tolist()
    Y_P = parties_df['AantalStemmen'].values.astype(float)

    n = len(Y_P)
    S_y = np.sum(Y_P)

    # 2. Define simulation parameters (error rates and transition probabilities)
    total = 10571990
    eps = 7766 / total

    eps_P = 0.95 * eps
    eps_B = 0.2 * (eps - eps_P)
    eps_INV = 0.8 * (eps - eps_P)

    zeta = 0.0005
    zeta_P = 0.8 * zeta
    zeta_INV = 0.2 * zeta

    eta = 0.005
    eta_P = 0.8 * eta
    eta_B = 0.2 * eta

    p = np.full(n, 0.5)
    p[0], p[-1] = 0.0, 1.0

    # 3. Perform matrix calculations for estimation and variance
    A_macro = np.array([[1 - eps_B - eps_INV, zeta_P, eta_P],
                        [eps_B, 1 - zeta, eta_B],
                        [eps_INV, zeta_INV, 1 - eta]])
    Y_macro = np.array([S_y, y_B, y_INV])
    X_macro_hat = np.linalg.inv(A_macro) @ Y_macro
    S_x_hat, x_B_hat, x_INV_hat = X_macro_hat

    C = (zeta_P * x_B_hat + eta_P * x_INV_hat) / S_x_hat
    M_hat = np.zeros((n, n))
    for j in range(n):
        M_hat[j, j] = 1 - eps + C
        if j > 0: M_hat[j - 1, j] = eps_P * p[j]
        if j < n - 1: M_hat[j + 1, j] = eps_P * (1 - p[j])

    X_P_hat = np.linalg.inv(M_hat) @ Y_P
    X_hat_full = np.concatenate([X_P_hat, [x_B_hat, x_INV_hat]])

    N = n + 2
    T_hat = np.zeros((N, N))
    w_hat = X_P_hat / S_x_hat
    for k in range(n):
        T_hat[k, k] = 1 - eps
        if k > 0: T_hat[k - 1, k] = eps_P * p[k]
        if k < n - 1: T_hat[k + 1, k] = eps_P * (1 - p[k])
        T_hat[n, k], T_hat[n + 1, k] = eps_B, eps_INV
    for i in range(n):
        T_hat[i, n], T_hat[i, n + 1] = zeta_P * w_hat[i], eta_P * w_hat[i]
    T_hat[n, n], T_hat[n + 1, n], T_hat[n, n + 1], T_hat[n + 1, n + 1] = 1 - zeta, zeta_INV, eta_B, 1 - eta

    Sigma_Y = np.zeros((N, N))
    for i in range(N):
        for m in range(N):
            if i == m:
                Sigma_Y[i, i] = sum(X_hat_full[k] * T_hat[i, k] * (1 - T_hat[i, k]) for k in range(N))
            else:
                Sigma_Y[i, m] = -sum(X_hat_full[k] * T_hat[i, k] * T_hat[m, k] for k in range(N))

    Sigma_X = np.linalg.inv(T_hat) @ Sigma_Y @ np.linalg.inv(T_hat).T
    SE = np.sqrt(np.diag(Sigma_X))

    # 4. Visualization of estimated bias and confidence intervals
    names_full = party_names + ['Blank', 'Invalid']
    bias = X_hat_full - np.concatenate([Y_P, [y_B, y_INV]])

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(names_full))[::-1]
    colors = ['#3498DB' if b >= 0 else '#E74C3C' for b in bias]

    bars = ax.barh(y_pos, bias, xerr=1.96 * SE, color=colors, alpha=0.7,
                   edgecolor='black', capsize=3, ecolor='black')

    ax.bar_label(bars, fmt='%.1f', padding=5, fontsize=16, color='black')

    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_full, fontsize=16)
    ax.set_xlabel('Number of Votes', fontsize=16)
    ax.set_title('Estimated Bias and 95% Confidence Intervals', fontsize=14, fontweight='bold')

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min * 1.15, x_max * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # 5. Output results table
    Y_full = np.concatenate([Y_P, [y_B, y_INV]])
    results_df = pd.DataFrame({
        'Party/Type': names_full,
        'Observed': Y_full,
        'Estimated': X_hat_full,
        'SE': SE,
        '95% CI Lower': X_hat_full - 1.96 * SE,
        '95% CI Upper': X_hat_full + 1.96 * SE,
        'Bias': bias
    })

    pd.set_option('display.float_format', '{:.2f}'.format)
    print("\n" + "=" * 80)
    print("DETERMINISTIC COUNTING ERROR SIMULATION RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == '__main__':
    analyze_voting_errors()