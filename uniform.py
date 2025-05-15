import matplotlib.pyplot as plt
import matplotlib
from da import generate_uniform_preferences, gale_shapley
matplotlib.use('Agg')

def experiment_1():
    ns = [10, 50, 100, 200, 500]
    trials = 5
    avg_proposals = []

    for n in ns:
        total = 0
        for _ in range(trials):
            doc_prefs, hosp_prefs = generate_uniform_preferences(n)
            _, num_proposals = gale_shapley(doc_prefs, hosp_prefs)
            total += num_proposals
        avg = total / trials
        avg_proposals.append(avg)
        print(f"n={n}, avg proposals: {avg:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ns, avg_proposals, marker='o', linestyle='-', color='royalblue')
    for i, val in enumerate(avg_proposals):
        plt.text(ns[i], val + 20, f"{val:.0f}", ha='center', fontsize=9)
    plt.title("Average Number of Proposals vs n", fontsize=14)
    plt.xlabel("n (Number of Doctors/Hospitals)", fontsize=12)
    plt.ylabel("Average Total Proposals", fontsize=12)
    plt.xticks(ns)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/exp1_avg_proposals_vs_n.png")


def experiment_2(n=200, trials=100):
    proposal_counts = []

    for _ in range(trials):
        doc_prefs, hosp_prefs = generate_uniform_preferences(n)
        _, p_count = gale_shapley(doc_prefs, hosp_prefs)
        proposal_counts.append(p_count)

    avg_proposals = sum(proposal_counts) / len(proposal_counts)

    plt.figure(figsize=(8, 5))
    plt.hist(proposal_counts, bins=20, edgecolor='black', color='lightskyblue')
    plt.axvline(avg_proposals, color='red', linestyle='--', label=f"Average = {avg_proposals:.1f}")
    plt.title(f"Proposal Count Distribution (n = {n})", fontsize=14)
    plt.xlabel("Total Proposals in One Run", fontsize=12)
    plt.ylabel("Frequency (out of 100 runs)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/exp2_proposal_distribution_n{n}.png")

def experiment_3():
    ns = [10, 50, 100, 200, 500]
    trials = 5
    doc_rank_list = []
    hosp_rank_list = []

    for n in ns:
        doc_total = 0
        hosp_total = 0
        for _ in range(trials):
            doc_prefs, hosp_prefs = generate_uniform_preferences(n)
            matches, _ = gale_shapley(doc_prefs, hosp_prefs)

            doctor_to_hosp = [None] * n
            for h, d in enumerate(matches):
                doctor_to_hosp[d] = h

            for d in range(n):
                h = doctor_to_hosp[d]
                doc_total += doc_prefs[d].index(h)

            for h in range(n):
                d = matches[h]
                hosp_total += hosp_prefs[h].index(d)

        avg_doc_rank = doc_total / (n * trials)
        avg_hosp_rank = hosp_total / (n * trials)
        doc_rank_list.append(avg_doc_rank)
        hosp_rank_list.append(avg_hosp_rank)

        print(f"n={n} -> Doctor Avg Rank: {avg_doc_rank:.2f}, Hospital Avg Rank: {avg_hosp_rank:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ns, doc_rank_list, marker='o', linestyle='-', color='forestgreen', label="Doctor Avg Rank")
    plt.plot(ns, hosp_rank_list, marker='s', linestyle='--', color='crimson', label="Hospital Avg Rank")
    plt.title("Average Rank of Matched Partner vs n", fontsize=14)
    plt.xlabel("n (Number of Doctors/Hospitals)", fontsize=12)
    plt.ylabel("Average Rank in Preference List", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(ns)
    plt.tight_layout()
    plt.savefig("plots/exp3_avg_rank_vs_n.png")

def experiment_4(n=200, trials=100):
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]  
    doc_counts = [0] * len(cutoffs)
    hosp_counts = [0] * len(cutoffs)
    total = n * trials

    for _ in range(trials):
        doc_prefs, hosp_prefs = generate_uniform_preferences(n)
        matches, _ = gale_shapley(doc_prefs, hosp_prefs)

        doctor_to_hosp = [None] * n
        for h, d in enumerate(matches):
            doctor_to_hosp[d] = h

        for d in range(n):
            h = doctor_to_hosp[d]
            rank = doc_prefs[d].index(h)
            for i, p in enumerate(cutoffs):
                if rank < int(n * p):
                    doc_counts[i] += 1

        for h in range(n):
            d = matches[h]
            rank = hosp_prefs[h].index(d)
            for i, p in enumerate(cutoffs):
                if rank < int(n * p):
                    hosp_counts[i] += 1

    doc_ratios = [x / total for x in doc_counts]
    hosp_ratios = [x / total for x in hosp_counts]
    x_percent = [int(p * 100) for p in cutoffs]

    plt.figure(figsize=(8, 5))
    plt.plot(x_percent, doc_ratios, marker='o', linestyle='-', color='blue', label="Doctors")
    plt.plot(x_percent, hosp_ratios, marker='s', linestyle='--', color='orange', label="Hospitals")
    plt.title(f"Proportion of Matches within Top p% Preferences (n = {n})", fontsize=14)
    plt.xlabel("Top p% of Preference List", fontsize=12)
    plt.ylabel("Proportion of Matches", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(x_percent)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"plots/exp4_top_percent_match_n{n}.png")


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
