"""Test how much the strategies reproduce the PrometheeII rankings."""
import promethee as prom
import data_reader as dr
from scipy import stats
import numpy


def compare(tests_qty=3):
    """Compare the different stratiegies."""
    output = "res/ReferencedPII/strategies/comparisons.txt"
    data_sets = ['EPI', 'SHA', 'GEQ']
    # data_sets = ['HDI']
    range_seed = range(0, 0 + tests_qty)
    alt_num = 30
    ref_number = 4
    strategies = [prom.strategy1, prom.strategy2,
                  prom.strategy3, prom.strategy4]
    # strategies = [prom.strategy2]

    kendall_taus = [[] for i in range(4)]           # One list for each strategy
    titles = []

    for data_set in data_sets:
        input_file = 'data/' + str(data_set) + '/raw.csv'
        alternatives = dr.open_raw(input_file)[0]

        for seed in range_seed:
            promethee = prom.PrometheeII(alternatives, seed=seed,
                                         alt_num=alt_num)
            prom_ranking = promethee.ranking

            title = data_set + str(seed)
            titles.append(title)

            for i, strategy in enumerate(strategies):
                referenced = prom.ReferencedPII(alternatives, seed=seed,
                                                strategy=strategy,
                                                alt_num=alt_num)
                refrank = referenced.ranking
                tau = stats.kendalltau(refrank, prom_ranking)[0]
                tau = int(tau*1000)/1000
                kendall_taus[i].append(tau)

    print_to_file(output, titles, kendall_taus, tests_qty)


def print_to_file(file_name, titles, kendall_taus, tests_qty):
    """Print results to file."""
    output = open(file_name, 'a')
    if (tests_qty < 10):
        output.write(str(titles) + '\n')
        for tau in kendall_taus:
            output.write(str(tau) + '\n')

    for i, tau in enumerate(kendall_taus):
        output.write('strategy : ' + str(i + 1) + '\n')
        output.write('  var = ' + str(numpy.var(tau)) + '\n')
        output.write('  mean = ' + str(numpy.mean(tau)) + '\n')
        output.write('  min = ' + str(min(tau)) + '\n')
