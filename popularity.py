import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from da import gale_shapley
matplotlib.use('Agg')

def generate_popularity_preferences(n, alpha=2.0):
    popularity = np.array([1 / (i + 1) ** alpha for i in range(n)])
    popularity /= popularity.sum()

    def weighted_shuffle(prob_dist):
        return list(np.random.choice(range(n), size=n, replace=False, p=prob_dist))

    doctors_prefs = [weighted_shuffle(popularity) for _ in range(n)]
    hospitals_prefs = [weighted_shuffle(popularity) for _ in range(n)]

    return doctors_prefs, hospitals_prefs

def experiment_pop1():
    ns = [10, 50, 100, 200, 500]
    trials = 5
    avg_proposals = []

    for n in ns:
        total = 0
        for _ in range(trials):
            d_prefs, h_prefs = generate_popularity_preferences(n)
            _, props = gale_shapley(d_prefs, h_prefs)
            total += props
        avg = total / trials
        avg_proposals.append(avg)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, avg_proposals, marker='o')
    plt.title("Average Number of Proposals vs n (Popularity Model)")
    plt.xlabel("n")
    plt.ylabel("Average Proposals")
    plt.yscale("log")  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/pop_exp1_avg_proposals_vs_n.png")


def experiment_pop2(n=200, trials=100):
    proposal_counts = []
    for _ in range(trials):
        d_prefs, h_prefs = generate_popularity_preferences(n)
        _, props = gale_shapley(d_prefs, h_prefs)
        proposal_counts.append(props)

    avg = sum(proposal_counts) / len(proposal_counts)

    plt.figure(figsize=(8, 5))
    plt.hist(proposal_counts, bins=15, color='skyblue', edgecolor='black')
    plt.axvline(avg, color='red', linestyle='--')
    plt.title(f'Proposal Count Distribution (n={n}) - Popularity Model')
    plt.xlabel('Total Proposals')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'plots/pop_exp2_proposal_distribution_n{n}.png')

def experiment_pop3():
    ns = [10, 50, 100, 200, 500]
    trials = 5
    doc_ranks = []
    hosp_ranks = []

    for n in ns:
        d_total, h_total = 0, 0
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
    plt.plot(ns, doc_ranks, label="Doctors", marker='o')
    plt.plot(ns, hosp_ranks, label="Hospitals", marker='s', linestyle='--')
    plt.title("Average Match Rank vs n (Popularity Model)")
    plt.xlabel("n")
    plt.ylabel("Avg Rank in Preference List")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/pop_exp3_avg_rank_vs_n.png")


def experiment_pop4(n=200, trials=100):
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
    plt.plot(x_vals, doc_rates, label="Doctors", marker='o')
    plt.plot(x_vals, hosp_rates, label="Hospitals", marker='s', linestyle='--')
    plt.title("Top p% Preference Match Rate (Popularity Model)")
    plt.xlabel("Top p%")
    plt.ylabel("Proportion Matched")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'plots/pop_exp4_top_percent_match_n{n}.png')


if __name__ == "__main__":
    experiment_pop1()
    experiment_pop2()
    experiment_pop3()
    experiment_pop4()
