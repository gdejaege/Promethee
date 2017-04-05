"""Test the number of references needed to discriminate all the alternatives."""
import promethee as prom
import data_reader as dr
import random


def count_draws(threshold):
    """Test with EPI, SHA, GEQ dataset."""
    data_sets = ['SHA', 'EPI', 'GEQ']
    output = "res/Referenced_PII_reference_quantity/thresh_" + str(threshold) \
        + ".txt"

    # Change these parameters if needed
    ref_numbers = [2, 3, 5, 10, 15, 25]
    alternative_numbers = [10, 20, 40, 80]
    seed_list = range(50)

    ref_set_strategy = prom.strategy1

    all_res = []
    for ref_number in ref_numbers:
        res = []
        for alt_number in alternative_numbers:
            tot = 0
            for seed in seed_list:
                for data_set in data_sets:
                    source = "data/" + data_set + "/raw.csv"
                    alts = dr.open_raw(source)[0]
                    ref_prom = prom.ReferencedPII(alts, alt_num=alt_number,
                                                  strategy=ref_set_strategy,
                                                  seed=seed, ref_num=ref_number)
                    tot += ref_prom.tied_ranking(ref_prom.scores, threshold)
            res.append(tot)
        all_res.append(res)
    print_to_file(output, ref_numbers, alternative_numbers, seed_list, all_res)


def print_to_file(file_name, ref_num, alt_num, seed_list, all_res):
    """Print to file."""
    template = "{:3d},"*(len(alt_num) - 1) + "{:3d}"

    output = open(file_name, 'a')
    output.write("ref_num: " + str(ref_num) + "\n")
    output.write("alt_num: " + str(alt_num) + "\n")
    output.write("seed_list: " + str(seed_list) + "\n")
    # Rank reversals info
    for row in all_res:
        output.write(template.format(*row))
        output.write("\n")
    output.write("\n"*2)
