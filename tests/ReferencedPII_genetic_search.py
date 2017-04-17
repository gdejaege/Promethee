"""Test the functions searching for reference profiles."""

import promethee as prom
import data_reader as dr
import genetic_references_search as GS
import time


def first_search():
    """Search good references sets for 15 random subsets of each data set."""
    data_sets = ['EPI', 'SHA', 'GEQ']
    weights, ceils = None, None
    seeds = range(15)

    alternative_numbers = [20, 25, 30, 40, 50]

    for data_set in data_sets:
        input_file = 'data/' + str(data_set) + '/raw.csv'
        output = 'res/ReferencedPII_genetic_search/' + str(data_set) + '.txt'
        alts = dr.open_raw(input_file)[0]
        for alt_num in alternative_numbers:
            succes = []
            failures = []
            failures_tau = []
            for s in seeds:
                t1 = time.time()
                tau = GS.genetic_search(alts, seed=s, weights=weights,
                                        ceils=ceils, alt_num=alt_num)
                print(str(s) + ', time: ' + str(time.time() - t1) + ', tau: '
                      + str(tau))
                if (tau > 1 - 1e-5):
                    succes.append(s)
                else:
                    failures.append(s)
                    tau_rounded = int(tau*1000)/1000
                    failures_tau.append(tau_rounded)
            save_res_to_file(output, alt_num, succes, failures, failures_tau)


def retry_failed(data_set, alt_numbers, failed_seeds, ref_number=5, maxrep=1):
    """Retry the search for subsets which failed the first time."""
    weights, ceils = None, None

    # Here we retry the seeds failed with different parameters
    t0 = time.time()
    alternative_numbers = alt_numbers
    seeds = failed_seeds
    input_file = 'data/' + str(data_set) + '/raw.csv'
    output = 'res/ReferencedPII_genetic_search/' + str(data_set) + '.txt'
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
                prob = 0.03 + 0.02*it
                tau2 = GS.genetic_search(alts, seed=s, weights=weights,
                                         RS_size=ref_number, ceils=ceils,
                                         alt_num=alt_num, pop_size=600,
                                         mut_prob=prob, MAXIT=50)
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
