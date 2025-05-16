import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from da import gale_shapley

matplotlib.use("Agg")

def _ensure_plot_dir(dirname="plots"):
    Path(dirname).mkdir(parents=True, exist_ok=True)

def weighted_shuffle(prob_dist):
    n = len(prob_dist)
    return list(np.random.choice(range(n), size=n, replace=False, p=prob_dist))

def generate_popularity_preferences(n, alpha=2.0):
    popularity = np.array([1 / (i + 1) ** alpha for i in range(n)], dtype=np.float64)
    popularity /= popularity.sum()

    doc_prefs = [weighted_shuffle(popularity) for _ in range(n)]
    hosp_prefs = [weighted_shuffle(popularity) for _ in range(n)]
    return doc_prefs, hosp_prefs

def generate_popularity_preferences_from_weights(n, weights):
    weights_arr = np.asarray(weights, dtype=np.float64)

    def shuffle_with_weights():
        noise = -np.log(np.random.uniform(size=n)) / weights_arr
        return list(np.argsort(noise))

    return [shuffle_with_weights() for _ in range(n)], [shuffle_with_weights() for _ in range(n)]

def avg_rank(d_prefs, h_prefs, matches):
    n = len(d_prefs)
    d_to_h = [None] * n
    for h, d in enumerate(matches):
        d_to_h[d] = h

    doc_rank = sum(d_prefs[d].index(d_to_h[d]) for d in range(n)) / n
    hosp_rank = sum(h_prefs[h].index(matches[h]) for h in range(n)) / n
    return doc_rank, hosp_rank

def experiment_pop1():
    _ensure_plot_dir()
    ns = [10, 50, 100, 200, 500]
    trials = 5

    avg_proposals = []
    for n in ns:
        total = 0
        for _ in range(trials):
            d_prefs, h_prefs = generate_popularity_preferences(n)
            _, props = gale_shapley(d_prefs, h_prefs)
            total += props
        avg_proposals.append(total / trials)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, avg_proposals, marker="o", color="tab:green")
    plt.title("Average Number of Proposals vs n (Popularity Model)")
    plt.xlabel("n")
    plt.ylabel("Average Proposals")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/pop_exp1_avg_proposals_vs_n.png")
    plt.close()

def experiment_pop2(n=200, trials=100):
    _ensure_plot_dir()
    proposal_counts = []
    for _ in range(trials):
        d_prefs, h_prefs = generate_popularity_preferences(n)
        _, props = gale_shapley(d_prefs, h_prefs)
        proposal_counts.append(props)

    avg = np.mean(proposal_counts)

    plt.figure(figsize=(8, 5))
    plt.hist(proposal_counts, bins=15, color="tab:orange", edgecolor="black")
    plt.axvline(avg, color="tab:red", linestyle="--")
    plt.title(f"Proposal Count Distribution (n={n}) - Popularity Model")
    plt.xlabel("Total Proposals")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/pop_exp2_proposal_distribution_n{n}.png")
    plt.close()

def experiment_pop3():
    _ensure_plot_dir()
    ns = [10, 50, 100, 200, 500]
    trials = 5

    doc_ranks = []
    hosp_ranks = []

    for n in ns:
        d_total = h_total = 0
        for _ in range(trials):
            d_prefs, h_prefs = generate_popularity_preferences(n)
            matches, _ = gale_shapley(d_prefs, h_prefs)
            d_to_h = [None] * n
            for h, d in enumerate(matches):
                d_to_h[d] = h
            d_total += sum(d_prefs[d].index(d_to_h[d]) for d in range(n))
            h_total += sum(h_prefs[h].index(matches[h]) for h in range(n))
        doc_ranks.append(d_total / (n * trials))
        hosp_ranks.append(h_total / (n * trials))

    plt.figure(figsize=(8, 5))
    plt.plot(ns, doc_ranks, label="Doctors", marker="o", color="tab:blue")
    plt.plot(ns, hosp_ranks, label="Hospitals", marker="s", linestyle="--", color="tab:red")
    plt.title("Average Match Rank vs n (Popularity Model)")
    plt.xlabel("n")
    plt.ylabel("Avg Rank in Preference List")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/pop_exp3_avg_rank_vs_n.png")
    plt.close()

def experiment_pop4(n=200, trials=100):
    _ensure_plot_dir()
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    doc_hits = [0] * len(cutoffs)
    hosp_hits = [0] * len(cutoffs)
    total = n * trials

    for _ in range(trials):
        d_prefs, h_prefs = generate_popularity_preferences(n)
        matches, _ = gale_shapley(d_prefs, h_prefs)
        d_to_h = [None] * n
        for h, d in enumerate(matches):
            d_to_h[d] = h

        for d in range(n):
            h = d_to_h[d]
            rank = d_prefs[d].index(h)
            for i, p in enumerate(cutoffs):
                if rank < int(n * p):
                    doc_hits[i] += 1

        for h in range(n):
            d = matches[h]
            rank = h_prefs[h].index(d)
            for i, p in enumerate(cutoffs):
                if rank < int(n * p):
                    hosp_hits[i] += 1

    x_vals = [int(p * 100) for p in cutoffs]
    doc_rates = [v / total for v in doc_hits]
    hosp_rates = [v / total for v in hosp_hits]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, doc_rates, label="Doctors", marker="o", color="tab:purple")
    plt.plot(x_vals, hosp_rates, label="Hospitals", marker="s", linestyle="--", color="tab:brown")
    plt.title("Top p% Preference Match Rate (Popularity Model)")
    plt.xlabel("Top p%")
    plt.ylabel("Proportion Matched")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/pop_exp4_top_percent_match_n{n}.png")
    plt.close()

def experiment_pop5(n=100, trials=100):
    _ensure_plot_dir()
    power_law_weights = [2 ** (i * 0.05) for i in range(n)]
    near_uniform_weights = np.linspace(1, 2, n)

    doc_ranks_power, hosp_ranks_power = [], []
    doc_ranks_uniform, hosp_ranks_uniform = [], []

    for _ in range(trials):
        d_prefs, h_prefs = generate_popularity_preferences_from_weights(n, power_law_weights)
        matches, _ = gale_shapley(d_prefs, h_prefs)
        d_rank, h_rank = avg_rank(d_prefs, h_prefs, matches)
        doc_ranks_power.append(d_rank)
        hosp_ranks_power.append(h_rank)

        d_prefs, h_prefs = generate_popularity_preferences_from_weights(n, near_uniform_weights)
        matches, _ = gale_shapley(d_prefs, h_prefs)
        d_rank, h_rank = avg_rank(d_prefs, h_prefs, matches)
        doc_ranks_uniform.append(d_rank)
        hosp_ranks_uniform.append(h_rank)

    plt.figure(figsize=(8, 5))
    plt.hist(doc_ranks_power, bins=20, alpha=0.5, label="Doctors – Power Law", color="tab:cyan", edgecolor="black")
    plt.hist(hosp_ranks_power, bins=20, alpha=0.5, label="Hospitals – Power Law", color="tab:olive", edgecolor="black")
    plt.hist(doc_ranks_uniform, bins=20, alpha=0.5, label="Doctors – Near Uniform", color="tab:pink", edgecolor="black")
    plt.hist(hosp_ranks_uniform, bins=20, alpha=0.5, label="Hospitals – Near Uniform", color="tab:gray", edgecolor="black")
    plt.title("Average Match Rank Distribution\nPower Law vs Near Uniform Popularity")
    plt.xlabel("Average Rank in Preference List")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/pop_exp5_rank_comparison_distributions.png")
    plt.close()

if __name__ == "__main__":
    experiment_pop1()
    experiment_pop2()
    experiment_pop3()
    experiment_pop4()
    experiment_pop5()
