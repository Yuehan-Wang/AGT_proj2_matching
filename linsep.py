import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from da import gale_shapley
import os

os.makedirs("plots", exist_ok=True)

def generate_linear_separable_preferences(n, lam=0.5):
    hospital_public = np.random.uniform(0, 1, n)
    doctor_public = np.random.uniform(0, 1, n)

    doctors_prefs = []
    for d in range(n):
        private_scores = np.random.uniform(0, 1, n)
        utilities = lam * hospital_public + (1 - lam) * private_scores
        prefs = np.argsort(-utilities) 
        doctors_prefs.append(prefs.tolist())

    hospitals_prefs = []
    for h in range(n):
        private_scores = np.random.uniform(0, 1, n)
        utilities = lam * doctor_public + (1 - lam) * private_scores
        prefs = np.argsort(-utilities)
        hospitals_prefs.append(prefs.tolist())

    return doctors_prefs, hospitals_prefs

def experiment_linsep1(lam=0.5):
    ns = [10, 50, 100, 200, 500]
    trials = 5
    avg_proposals = []

    for n in ns:
        total = 0
        for _ in range(trials):
            d_prefs, h_prefs = generate_linear_separable_preferences(n, lam)
            _, props = gale_shapley(d_prefs, h_prefs)
            total += props
        avg = total / trials
        avg_proposals.append(avg)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, avg_proposals, marker='o', color='royalblue')
    for i, val in enumerate(avg_proposals):
        plt.text(ns[i], val * 1.1, f"{int(val)}", ha='center', fontsize=9)
    plt.title(f"Average Proposals vs n (λ = {lam})")
    plt.xlabel("n")
    plt.ylabel("Average Proposals")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/linsep_exp1_avg_proposals_vs_n_lambda{int(lam*10)}.png")


def experiment_linsep2(n=200, trials=100, lam=0.5):
    proposal_counts = []
    for _ in range(trials):
        d_prefs, h_prefs = generate_linear_separable_preferences(n, lam)
        _, props = gale_shapley(d_prefs, h_prefs)
        proposal_counts.append(props)

    avg = sum(proposal_counts) / len(proposal_counts)

    plt.figure(figsize=(8, 5))
    plt.hist(proposal_counts, bins=12, color='lightskyblue', edgecolor='black', alpha=0.85)
    plt.axvline(avg, color='crimson', linestyle='--', linewidth=2, label=f"Avg = {avg:.1f}")
    plt.title(f"Proposal Distribution (n={n}, λ = {lam})")
    plt.xlabel("Total Proposals")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"plots/linsep_exp2_proposal_distribution_n{n}_lambda{int(lam*10)}.png")


def experiment_linsep3(lam=0.5):
    ns = [10, 50, 100, 200, 500]
    trials = 5
    doc_ranks = []
    hosp_ranks = []

    for n in ns:
        d_total, h_total = 0, 0
        for _ in range(trials):
            d_prefs, h_prefs = generate_linear_separable_preferences(n, lam)
            matches, _ = gale_shapley(d_prefs, h_prefs)
            d_to_h = [None] * n
            for h, d in enumerate(matches):
                d_to_h[d] = h
            d_total += sum(d_prefs[d].index(d_to_h[d]) for d in range(n))
            h_total += sum(h_prefs[h].index(matches[h]) for h in range(n))

        doc_ranks.append(d_total / (n * trials))
        hosp_ranks.append(h_total / (n * trials))

    plt.figure(figsize=(8, 5))
    plt.plot(ns, doc_ranks, label="Doctors", marker='o', color='blue')
    plt.plot(ns, hosp_ranks, label="Hospitals", marker='s', linestyle='--', color='darkorange')
    plt.title(f"Avg Match Rank vs n (λ = {lam})")
    plt.xlabel("n")
    plt.ylabel("Average Rank")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/linsep_exp3_avg_rank_vs_n_lambda{int(lam*10)}.png")

def experiment_linsep4(n=200, trials=100, lam=0.5):
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    doc_hits = [0] * len(cutoffs)
    hosp_hits = [0] * len(cutoffs)
    total = n * trials

    for _ in range(trials):
        d_prefs, h_prefs = generate_linear_separable_preferences(n, lam)
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
    plt.plot(x_vals, doc_rates, label="Doctors", marker='o', color='green')
    plt.plot(x_vals, hosp_rates, label="Hospitals", marker='s', linestyle='--', color='red')
    plt.title(f"Top p% Match Rate (n={n}, λ = {lam})")
    plt.xlabel("Top p%")
    plt.ylabel("Proportion Matched")
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"plots/linsep_exp4_top_percent_match_n{n}_lambda{int(lam*10)}.png")

def compute_private_utilities(n=200, trials=100, lam=0.5):
    doctor_scores = []
    hospital_scores = []

    for _ in range(trials):
        hospital_public = np.random.uniform(0, 1, n)
        doctor_public = np.random.uniform(0, 1, n)

        doctor_private = np.random.uniform(0, 1, (n, n))  
        hospital_private = np.random.uniform(0, 1, (n, n))  

        doctors_prefs = []
        hospitals_prefs = []

        for d in range(n):
            utilities = lam * hospital_public + (1 - lam) * doctor_private[d]
            prefs = np.argsort(-utilities)
            doctors_prefs.append(prefs.tolist())

        for h in range(n):
            utilities = lam * doctor_public + (1 - lam) * hospital_private[h]
            prefs = np.argsort(-utilities)
            hospitals_prefs.append(prefs.tolist())


        matches, _ = gale_shapley(doctors_prefs, hospitals_prefs)

        doc_private_util = 0
        hosp_private_util = 0

        for h, d in enumerate(matches):
            doc_private_util += doctor_private[d][h]
            hosp_private_util += hospital_private[h][d]

        doctor_scores.append(doc_private_util / n)
        hospital_scores.append(hosp_private_util / n)

    avg_doc_score = sum(doctor_scores) / trials
    avg_hosp_score = sum(hospital_scores) / trials

    print(f"[λ = {lam}] Avg doctor private utility: {avg_doc_score:.4f}")
    print(f"[λ = {lam}] Avg hospital private utility: {avg_hosp_score:.4f}")

if __name__ == "__main__":
    lam = 0.5
    experiment_linsep1(lam)
    experiment_linsep2(lam=lam)
    experiment_linsep3(lam)
    experiment_linsep4(lam=lam)
    compute_private_utilities(n=200, trials=100, lam=lam)
