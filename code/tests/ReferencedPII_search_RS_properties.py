"""Test file of the correct set of reference profiles found."""

import promethee as prom
import data_reader as dr
import adaptive_questioning_procedure as aqp

import os
import string
import time
from scipy import stats
import numpy


def RS_from_aqp(data_set="GEQ", seeds=range(3), alt_num=20):
    """Analyse the correct RS found with this procedure."""
    alts_file_name = "data/" + data_set + "/raw.csv"
    all_alts = dr.open_raw(alts_file_name)[0]

    mean_mean_ratio_str = []
    var_var_ratio_str = []
    mean_var_ratio_str = []
    var_mean_ratio_str = []

    template_ratio = '{0:^d}|'
    for i in range(len(all_alts[0])):
        template_ratio += '{' + str(i+1) + ':+.3F}|'

    # Output
    output_file = "res/ReferencedPII_RS_analysis/" + data_set

    for seed in seeds:
        # Input
        RS_prefix = "res/ReferencedPII_questioning_procedure/"
        all_RS_file_name = data_set + "/" + str(seed) + ".csv"
        all_RS = dr.open_raw_RS(RS_prefix + all_RS_file_name)

        # get the correct alt_num for the concerned seed
        promethee = prom.PrometheeII(all_alts, seed=seed, alt_num=alt_num)
        alts_per_criterion = list(map(list, zip(*promethee.alternatives)))

        # Check if the parameters (= alternative subset) are indeed the same
        questioning_procedure = aqp.Adaptive_procedure(all_alts, seed=seed,
                                                       alt_num=alt_num,
                                                       ref_number=4,
                                                       pts_per_random_it=200,
                                                       desired_points=3000)
        if (not prom.check_parameters(questioning_procedure.promethee,
                                      promethee)):
            print("error")

        """Will contain lists of means of the ref's evaluation for each criterion
        ex:
            all_means_ratio[0] = [mean(c1(r1), ..., mean(c2(r1), ..., c2(r4))]
            RS_means[2] = [...]
        """
        # List of all ratios for individual RS
        all_mean_ratios = []
        all_var_ratios = []

        for i in range(len(all_RS)):
            RS = all_RS[i]

            # matrix = list of criteria which are lists of refs or
            # alternatives evaluations
            refs_per_criterion = list(map(list, zip(*RS)))

            # ratio between estimator of on RS compared to the one of the alts
            individual_mean_ratios, individual_var_ratios = [], []
            for crit in range(len(refs_per_criterion)):
                var_ref = numpy.var(refs_per_criterion[crit])
                mean_ref = numpy.mean(refs_per_criterion[crit])
                var_alt = numpy.var(alts_per_criterion[crit])
                mean_alt = numpy.mean(alts_per_criterion[crit])

                individual_mean_ratios.append(mean_ref/mean_alt)
                individual_var_ratios.append(var_ref/var_alt)

            all_mean_ratios.append(individual_mean_ratios)
            all_var_ratios.append(individual_var_ratios)

        # transpose the matrix : a list of references sets which are lists
        # of the estimators for each criterion becomes a list of estimators for
        # each criterion which contains the estimater for each RS
        var_ratios_per_crit = list(map(list, zip(*all_var_ratios)))
        mean_ratios_per_crit = list(map(list, zip(*all_mean_ratios)))

        var_var_ratios = [numpy.var(crit) for crit in var_ratios_per_crit]
        mean_var_ratios = [numpy.mean(crit) for crit in var_ratios_per_crit]
        var_mean_ratios = [numpy.var(crit) for crit in mean_ratios_per_crit]
        mean_mean_ratios = [numpy.mean(crit) for crit in mean_ratios_per_crit]

        # Transorm in strings
        var_var_ratio_str.append(template_ratio.format(seed, *var_var_ratios))
        var_mean_ratio_str.append(template_ratio.format(seed, *var_mean_ratios))
        mean_var_ratio_str.append(template_ratio.format(seed, *mean_var_ratios))
        mean_mean_ratio_str.append(template_ratio.format(seed,
                                                         *mean_mean_ratios))

    with open(output_file, 'a') as output:
        output.write("var(var(ref)/var(alt)) \n")
        for i in var_var_ratio_str:
            output.write(i)
            output.write("\n")
        output.write("\n")

        output.write("var(mean(ref)/mean(alt)) \n")
        for i in var_mean_ratio_str:
            output.write(i)
            output.write("\n")
        output.write("\n")

        output.write("mean(var(ref)/var(alt)) \n")
        for i in mean_var_ratio_str:
            output.write(i)
            output.write("\n")
        output.write("\n")

        output.write("mean(mean(ref)/mean(alt)) \n")
        for i in mean_mean_ratio_str:
            output.write(i)
            output.write("\n")
        output.write("\n")
