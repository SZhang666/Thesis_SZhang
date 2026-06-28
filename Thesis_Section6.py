import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_voter_uncertainty(filepath='vote_bloc.csv', M=1000):
    """Run Monte Carlo simulation for voter uncertainty."""
    # Load and clean data
    try:
        df = pd.read_csv(filepath, sep=';')
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None, None

    n_parties = len(df)
    initial_votes = df['AantalStemmen'].values
    parties = df['Partij'].values
    blocs = df['New_Bloc'].values
    bloc_names = ['B1', 'B2', 'B3', 'B4', 'B5']

    # Configuration for empirical baseline scenario
    params = {
        'alpha': {'B1': 0.20, 'B2': 0.35, 'B3': 0.40, 'B4': 0.25, 'B5': 0.15},
        'beta': {'B1': 0.30, 'B2': 0.45, 'B3': 0.50, 'B4': 0.40, 'B5': 0.10},
        'p': {'B1': 0.00, 'B2': 0.60, 'B3': 0.40, 'B4': 0.20, 'B5': 1.00}
    }

    # Calculate relative weights and transition matrix A
    df['omega'] = df.groupby('New_Bloc')['AantalStemmen'].transform(lambda x: x / x.sum())
    omega = df['omega'].values

    A = np.zeros((n_parties, n_parties))
    for i in range(n_parties):
        for j in range(n_parties):
            b_i, b_j = blocs[i], blocs[j]
            a_i, b_i_param = params['alpha'][b_i], params['beta'][b_i]

            if i == j:
                A[i, j] = 1 - a_i
            elif b_i == b_j:
                denominator = 1 - omega[i]
                A[i, j] = a_i * (1 - b_i_param) * (omega[j] / denominator if denominator > 0 else 0)
            else:
                b_i_idx, b_j_idx = int(b_i[1]), int(b_j[1])
                if abs(b_i_idx - b_j_idx) == 1:
                    direction_p = params['p'][b_i] if b_j_idx < b_i_idx else (1 - params['p'][b_i])
                    A[i, j] = direction_p * a_i * b_i_param * omega[j] * 2

    A = np.maximum(A, 0)

    # Perform Monte Carlo simulation
    results = np.zeros((M, n_parties))
    bloc_results_M = np.zeros((M, len(bloc_names)))

    for r in range(M):
        Z = np.zeros((n_parties, n_parties))
        for i in range(n_parties):
            prob_vector = A[i, :] / A[i, :].sum()
            Z[i, :] = np.random.multinomial(initial_votes[i], prob_vector)

        results[r, :] = Z.sum(axis=0)
        for idx, b_name in enumerate(bloc_names):
            bloc_results_M[r, idx] = results[r, blocs == b_name].sum()

    # Calculate statistics and 95% Confidence Intervals
    party_ci_lower = np.percentile(results, 2.5, axis=0)
    party_ci_upper = np.percentile(results, 97.5, axis=0)
    bloc_ci_lower = np.percentile(bloc_results_M, 2.5, axis=0)
    bloc_ci_upper = np.percentile(bloc_results_M, 97.5, axis=0)

    party_results = pd.DataFrame({
        'Party': parties, 'Bloc': blocs, 'Original': initial_votes,
        'Simulated_Mean': results.mean(axis=0),
        'CI_Lower_95': party_ci_lower, 'CI_Upper_95': party_ci_upper
    })
    party_results['Difference'] = party_results['Simulated_Mean'] - party_results['Original']

    orig_bloc_votes = df.groupby('New_Bloc')['AantalStemmen'].sum().reindex(bloc_names)
    bloc_results = pd.DataFrame({
        'Bloc': bloc_names, 'Original': orig_bloc_votes.values,
        'Simulated_Mean': bloc_results_M.mean(axis=0),
        'CI_Lower_95': bloc_ci_lower, 'CI_Upper_95': bloc_ci_upper
    })
    bloc_results['Difference'] = bloc_results['Simulated_Mean'] - bloc_results['Original']

    return party_results, bloc_results


def plot_monte_carlo_bias(party_results):
    """Visualize per-party vote bias and confidence intervals."""
    parties = party_results['Party'].tolist()
    bias = party_results['Difference'].values
    lower_error = party_results['Simulated_Mean'].values - party_results['CI_Lower_95'].values
    upper_error = party_results['CI_Upper_95'].values - party_results['Simulated_Mean'].values
    asymmetric_error = np.array([lower_error, upper_error])

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(parties))[::-1]
    colors = ['#3498DB' if b >= 0 else '#E74C3C' for b in bias]

    bars = ax.barh(y_pos, bias, xerr=asymmetric_error, color=colors, alpha=0.7,
                   edgecolor='black', capsize=3, ecolor='black')

    ax.bar_label(bars, fmt='%.1f', padding=5, fontsize=16)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parties, fontsize=16)
    ax.set_xlabel('Number of Votes', fontsize=16)
    ax.set_title(r'Average Bias $\bar{\Delta} = \bar{\mathcal{Z}}^{\prime} - Y$ per Party', fontsize=14,
                 fontweight='bold')

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min * 1.15, x_max * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_monte_carlo_bloc_bias(bloc_results):
    """Visualize per-bloc vote bias and confidence intervals."""
    blocs = bloc_results['Bloc'].tolist()
    bias = bloc_results['Difference'].values
    lower_error = bloc_results['Simulated_Mean'].values - bloc_results['CI_Lower_95'].values
    upper_error = bloc_results['CI_Upper_95'].values - bloc_results['Simulated_Mean'].values
    asymmetric_error = np.array([lower_error, upper_error])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(blocs))[::-1]
    colors = ['#3498DB' if b >= 0 else '#E74C3C' for b in bias]

    bars = ax.barh(y_pos, bias, xerr=asymmetric_error, color=colors, alpha=0.7,
                   edgecolor='black', capsize=5, ecolor='black')

    ax.bar_label(bars, fmt='%.1f', padding=5, fontsize=16)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(blocs, fontsize=16)
    ax.set_xlabel('Number of Votes', fontsize=16)
    ax.set_title(r'Average Bias $\bar{\Delta} = \bar{\mathcal{Z}}^{\prime} - Y$ per Bloc', fontsize=14,
                 fontweight='bold')

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min * 1.15, x_max * 1.15)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def main():
    print("Running Monte Carlo simulation (M=1000)...")
    party_results, bloc_results = simulate_voter_uncertainty('vote_bloc.csv', M=1000)

    if party_results is not None:
        pd.set_option('display.float_format', '{:,.0f}'.format)
        print("\n--- Party-level Summary ---")
        print(party_results)
        print("\n--- Bloc-level Summary ---")
        print(bloc_results)

        plot_monte_carlo_bias(party_results)
        plot_monte_carlo_bloc_bias(bloc_results)


if __name__ == '__main__':
    main()