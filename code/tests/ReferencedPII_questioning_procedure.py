"""Test the adaptive questioning procedure."""

import promethee as prom
import data_reader as dr
import adaptive_questioning_procedure as aqp

from contextlib import redirect_stdout
import csv
import random
import sys
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages


def analyse(alt_num=20, seeds=range(0,3), data_sets=['EPI', 'SHA', 'GEQ'],
            rounds=20, make_pdf=False):
    """Analyse the results of the adaptive questioning procedure."""
    weights, ceils = None, None
    seeds = range(3, 4)

    output_dir = 'res/ReferencedPII/adaptive_questioning_procedure/'
    output_file = open(output_dir + "adaptative_questionning_results2.txt", "a")
    # pp = PdfPages(output_dir + 'kendall_tau_boxplots.pdf')

    for data_set in data_sets:
        input_file = 'data/' + str(data_set) + '/raw.csv'
        alts = dr.open_raw(input_file)[0]
        for seed in seeds:
            correct_pts_output = ('res/ReferencedPII_questioning_procedure/'
                                  + data_set + '/' + str(seed) + '.csv')
            title = data_set + ' with ' + str(alt_num) + ' alternatives (seed '\
                + str(seed) + ')'
            title_plot = ('Adaptive questioning procedure on a subset of the '
                          + data_set + ' data set with ' + str(alt_num)
                          + ' alternatives')
            print(title)
            if True:
            # with redirect_stdout(output_file):
                print(title)
                procedure = aqp.Adaptive_procedure(alts, seed=seed,
                                                   alt_num=alt_num,
                                                   pts_per_random_it=200,
                                                   desired_points=3000)
                corrects = procedure.execute(rounds)
                write_correct_pts(corrects, correct_pts_output)
                print()
            if (make_pdf):
                # Boxplot of the rankings
                fig = plt.figure(1, figsize=(9, 6))
                plt.suptitle(title_plot)
                ax = fig.add_subplot(111)
                ax.set_ylim(-0.3, 1.1)
                ax.yaxis.set_major_locator(ticker.FixedLocator([-0.25, 0,
                                                                0.25, 0.5,
                                                                0.75, 1]))
                bp = ax.boxplot(procedure.kendall_taus)
                # pp.savefig(bbox_inches='tight')
                fig.savefig(output_dir + title + '.pdf', bbox_inches='tight')
                plt.clf()
    output_file.close()
    # pp.close()

def write_correct_pts(pts, output_file, qty=100):
    """Write correct points to file."""
    pts_qty = len(pts)
    qty = min(qty, pts_qty)

    if qty > 0:
        output = open(output_file, 'a')
        wr = csv.writer(output, delimiter=',')
        points_to_write = random.sample(pts, qty)
        for RS in points_to_write:
            for ref in RS:
                wr.writerow(ref)
            wr.writerow(['##########'])


def test_functions():
    """Test various functions of the procedure."""
    data_set = 'EPI'
    weights, ceils = None, None
    seed = 0
    res = True

    input_file = 'data/' + str(data_set) + '/raw.csv'
    alts = dr.open_raw(input_file)[0]
    procedure = aqp.Adaptive_procedure(alts, seed=seed, alt_num=10,
                                       ref_number=4,
                                       pts_per_random_it=2,
                                       desired_points=10)

    # Constraint verification
    procedure.add_constraint((6, 8))
    procedure.add_constraint((7, 8))
    procedure.add_constraint((3, 5))
    if (procedure.is_admissible([1, 2, 9, 4, 5, 3, 6, 7, 8])):
        res = False
    if (not procedure.is_admissible([1, 2, 3, 4, 5, 9, 6, 7, 8])):
        res = False
    print(res)
