"""Test how much the strategies reproduce the PrometheeII rankings."""
import promethee as prom
import data_reader as dr
from scipy import stats


def compare():
    """Test the strategies."""
    data_sets = ['EPI', 'SHA', 'GEQ']
    # data_sets = ['HDI']
    range_seed = range(80,82)
    alt_num = 30
    ref_number = 4
    strategies = [prom.strategy1, prom.strategy2, 
                  prom.strategy3, prom.strategy4]
    # strategies = [prom.strategy2]

    kendall_taus = [[] for i in range(4)]           # One list for each strategy
    titles = []
    output = 'res/ReferencedPII_strategies_comparison/results.txt'
    

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
                kendall_taus[i].append(tau)

    for tau in kendall_taus:
        print(tau)



