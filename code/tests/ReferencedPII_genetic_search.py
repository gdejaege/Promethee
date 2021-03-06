"""Test the functions searching for reference profiles."""

import promethee as prom
import data_reader as dr
import genetic_references_search as GS
import time


def first_search(pop_size=600, mut_prob=0.01, MAXIT=50):
    """Try to find sets of reference profiles reproducing th PII ranking.

    Search for 15 different seeds. Once some positive results have been found,
    please use the next function to try again seeds that failed.
    """
    data_sets = ['SHA', 'EPI', 'GEQ']
    weights, ceils = None, None
    seeds = range(15)

    alternative_numbers = [20, 25, 30, 40, 50]

    for data_set in data_sets:
        input_file = 'data/' + str(data_set) + '/raw.csv'
        output = 'res/ReferencedPII/genetic_search/' + str(data_set) + '.txt'
        alts = dr.open_raw(input_file)[0]
        for alt_num in alternative_numbers:
            succes = []
            failures = []
            failures_tau = []
            for s in seeds:
                t1 = time.time()
                tau = GS.genetic_search(alts, seed=s, weights=weights,
                                        ceils=ceils, alt_num=alt_num,
                                        pop_size=pop_size, mut_prob=mut_prob,
                                        MAXIT=MAXIT)
                print(str(s) + ', time: ' + str(time.time() - t1) + ', tau: '
                      + str(tau))
                if (tau > 1 - 1e-5):
                    succes.append(s)
                else:
                    failures.append(s)
                    tau_rounded = int(tau*1000)/1000
                    failures_tau.append(tau_rounded)
            save_res_to_file(output, alt_num, succes, failures, failures_tau)


def retry_failed(data_set='SHA', alt_numbers=[20], failed_seeds=[[7, 8]],
                 ref_number=5, maxrep=1, pop_size=600, mut_prob=0.01, MAXIT=50):
    """Retry the search for subsets which failed the first time."""
    weights, ceils = None, None

    # Here we retry the seeds failed with different parameters
    t0 = time.time()
    alternative_numbers = alt_numbers
    seeds = failed_seeds
    input_file = 'data/' + str(data_set) + '/raw.csv'
    output = 'res/ReferencedPII/genetic_search/' + str(data_set) + '.txt'
    alts = dr.open_raw(input_file)[0]
    for i, alt_num in enumerate(alternative_numbers):
        succes = []
        failures = []
        failures_tau = []
        for s in seeds[i]:
            t1 = time.time()
            tau = 0
            it = 0
            while (tau < 1 - 1e-5 and it < maxrep):
                tau2 = GS.genetic_search(alts, seed=s, weights=weights,
                                         SRP_size=ref_number, ceils=ceils,
                                         alt_num=alt_num, pop_size=pop_size,
                                         mut_prob=mut_prob, MAXIT=MAXIT)
                tau = max(tau, tau2)
                print(str(s) + ', total time: ' + str(time.time() - t0) +
                      ", it time: " + str(time.time() - t1) + ', tau: '
                      + str(tau))
                it += 1
            if (tau > 1 - 1e-5):
                succes.append(s)
            else:
                failures.append(s)
                tau_rounded = int(tau*1000)/1000
                failures_tau.append(tau_rounded)
        save_res_to_file(output, alt_num, succes, failures, failures_tau)
    print("time :" + str(time.time() - t1))


def save_res_to_file(file_name, alt_num, succes, failures, failures_tau):
    """Print the results in a file."""
    output = open(file_name, 'a')
    output.write('\n')
    title = '#'*5 + ' ' + str(alt_num) + 'alts ' + '#'*5 + '\n'
    output.write(title)
    output.write("succesfull seeds: " + str(succes) + "\n")
    output.write("failed  seeds: " + str(failures) + "\n")
    output.write("failed taus: " + str(failures_tau) + "\n")
    output.write("\n"*3)
