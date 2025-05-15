import random

def generate_uniform_preferences(n):
    doctors_prefs = []
    hospitals_prefs = []
    for _ in range(n):
        doctor_pref = list(range(n))
        hospital_pref = list(range(n))
        random.shuffle(doctor_pref)
        random.shuffle(hospital_pref)
        doctors_prefs.append(doctor_pref)
        hospitals_prefs.append(hospital_pref)
    return doctors_prefs, hospitals_prefs

def gale_shapley(doctors_prefs, hospitals_prefs):
    n = len(doctors_prefs)
    free_doctors = list(range(n))
    proposals = [0] * n
    matches = [None] * n
    inverse_hospital_prefs = [None] * n

    for h in range(n):
        inverse_hospital_prefs[h] = {doc: rank for rank, doc in enumerate(hospitals_prefs[h])}

    proposal_count = 0
    while free_doctors:
        d = free_doctors.pop(0)
        h = doctors_prefs[d][proposals[d]]
        proposals[d] += 1
        proposal_count += 1

        if matches[h] is None:
            matches[h] = d
        else:
            current = matches[h]
            if inverse_hospital_prefs[h][d] < inverse_hospital_prefs[h][current]:
                matches[h] = d
                free_doctors.append(current)
            else:
                free_doctors.append(d)

    return matches, proposal_count
