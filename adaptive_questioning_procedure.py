"""Rough adaptative questionning procedure to find RS reproducing PII ranks.

The procedure is based on random points generation in a ref_num*crit_num
dimension space. Each point represent a reference set (RS).

At each iteration, the ranking obtained with the most admissible points is used
to compute the query candidates which are couple of pairwise succesive
alternatives.

One couple is selected and the DM will have to decide which alternatives
is the one he prefers (this is done according to the PII ranking that we are
trying to reproduce). This will yield a new constraint and some points will not
be admissible anymore.

For performence purposes, some reduced promethee methods are re-implemented
here.
"""

import promethee as PII
import data_reader as dr

from contextlib import redirect_stdout
import sys
import string
import random
import numpy as np
import scipy.spatial as spp
import scipy.stats as stats
import time

import os


class Adaptive_procedure:

    """Adaptive procedure to find possible RS reproducing PII rankings."""

    def __init__(self, init_alternatives, seed=0, alt_num=30, ref_number=4,
                 pts_per_random_it=200, random_add_it=1500, divide_it=5,
                 desired_points=3000):
        """Constructor.

        Inputs:
            init_alternatives - matrix composed of one list of evaluations for
                                each alternative.
            seed - used to generate some pseudo random parameters.
            max_alt - maximal number of alternatives on which the procedure must
                      be applied.
            ref_number - number of reference profiles in each set.
            pts_per_random_it - minimal quantity of points which are tried to
                                be added at random 'simultaneously'. This
                                quantity is repeated 'random_add_it' times at
                                each iteration of the procedure.
            random_add_it - quantity of times at each iteration of the procedure
                            'pts_per_random_it' are considered to be added to
                            the set of all admissible points.
            divide_it - number of times we try to add a new point near of an
                        admissible one (for each of the admissible ones).
            desired_points - desired size of the set of admissible points after
                             each iteration.

            These four last arguments are used because it is computationally
            not possible to start with a big enough set of admissible points.
            Therefore, at each iteration some points. More information in the
            'round_add_points' function.
        """
        self.ref_number = ref_number
        self.pts_per_random_it = pts_per_random_it
        self.desired_points = desired_points
        self.seed = seed
        self.promethee = PII.PrometheeII(init_alternatives, seed=self.seed,
                                         alt_num=alt_num)
        self.PII_ranking = self.promethee.ranking
        self.alternatives = self.promethee.alternatives

        # Used to add new points
        self.min_per_crit = [min(crit) for crit in self.promethee.eval_per_crit]
        self.max_per_crit = [max(crit) for crit in self.promethee.eval_per_crit]
        self.delta_per_crit = [self.max_per_crit[crit] - self.min_per_crit[crit]
                               for crit in range(len(self.max_per_crit))]

        self.crit_number = len(self.promethee.alternatives[0])

        # RS only used to initialise
        RS = [[1 for i in range(self.crit_number)] for ref in range(ref_number)]
        self.referenced = PII.ReferencedPII(init_alternatives, seed=self.seed,
                                            alt_num=alt_num,
                                            ref_set=RS)

        if (not PII.check_parameters(self.promethee, self.referenced)):
            print('parameters not equal between method')
            exit()

        self.admissible_points = []
        self.correct_points = []
        self.constraints = []

        # Matrix that keep trace of all the rankings (one row per iteration)
        self.kendall_taus = []

        self.add_initial_points()
        # self.compute_center()
        # define the template for printing the points
        self.it_template = "{:^3d}|{: ^9d}|{: ^10d}|" \
            + "{:^7d}|{: ^7.3f}|{: ^7.3f}|{: ^7.3f}|{: ^7.3f}|{: ^10s}|{: ^9d}"
        self.iteration = 0

    def add_initial_points(self):
        """Add the initial points."""
        tot_points = len(self.admissible_points) + len(self.correct_points)
        while (tot_points < self.desired_points):
            self.add_random_points()
            tot_points = len(self.admissible_points) + len(self.correct_points)

    def add_random_points(self):
        """try to add random points that satisfy all the constraints.

        The points are in ref_num*crit_num dimension spaces.
        """
        for point_iteration in range(self.pts_per_random_it):
            pt = [[random.uniform(
                self.min_per_crit[j] - 0.5*self.delta_per_crit[j],
                self.max_per_crit[j] + 0.5*self.delta_per_crit[j])
                   for j in range(self.crit_number)]
                  for k in range(self.ref_number)]

            refflows = self.referenced.compute_scores(ref_set=pt)
            ranking = self.referenced.compute_ranking(refflows)
            if (ranking == self.PII_ranking):
                self.correct_points.append(pt)
            elif (self.is_admissible(ranking)):
                self.admissible_points.append(pt)

    def divide_points(self):
        """Create new points on basis of the existing ones."""
        all_points = self.admissible_points + self.correct_points

        for pt in all_points:
            # Transposition: list[references][crit] -> list[crit][reference]
            RS_per_crit = list(map(list, zip(*pt)))

            for crit in range(len(RS_per_crit)):
                # Create a new point which can not be further than 5% of the
                # maximal difference of each criterion from the the cloned point
                lim = self.delta_per_crit[crit]*0.05
                RS_per_crit[crit] = \
                    [el + random.uniform(-lim, lim) for el in RS_per_crit[crit]]
            point = list(map(list, zip(*RS_per_crit)))

            refflows = self.referenced.compute_scores(ref_set=point)
            ranking = self.referenced.compute_ranking(refflows)
            if (ranking == self.PII_ranking):
                self.correct_points.append(pt)
            elif (self.is_admissible(ranking)):
                self.admissible_points.append(point)

    def is_admissible(self, ranking):
        """Check if the ranking satisfy all the constraints.

        Returns a boolean
        """
        for c in self.constraints:
            if (ranking.index(c[0]) > ranking.index(c[1])):
                return False
        return True

    def compute_center(self):
        """Compute the center of the points.

        This center will be used to make the next query. The ref-ranking will
        be computed at this center, then the alpha pair of alternatives with
        the smallest ref-flow difference will be envisaged as next query.
        """
        center = [[0 for i in range(self.crit_number)]
                  for j in range(self.ref_number)]
        all_pts = self.admissible_points + self.correct_points
        number_pts = len(all_pts)
        for pt in all_pts:
            for ref in range(self.ref_number):
                for crit in range(self.crit_number):
                    center[ref][crit] += pt[ref][crit]
        center = [[crit/number_pts for crit in center[ref]]
                  for ref in range(len(center))]
        self.center = center

    def query_cadidates(self):
        """Return query candidates according to the most commonest ranking.

        query candidates are pairs of succesive alternatives in this ranking
        """
        ref_ranking = self.commonest_ranking

        candidates = []
        for i in range(len(ref_ranking)-1):
            ai = ref_ranking[i]
            aj = ref_ranking[i+1]
            if self.is_admissible_constraint((ai, aj)):
                candidates.append((ai, aj))

        if(len(candidates) == 0):
            print(ref_ranking)
            print(self.constraints)
            exit("no admissible candidate")
        return candidates

    def select_best_candidate(self, candidates):
        """Return the candidate with the highest discriminationg power."""
        best_score = -1
        for cand in candidates:
            score = self.eval_candidate(cand)
            if score > best_score:
                best_score = score
                best_cand = cand
        return best_cand

    def eval_candidate(self, candidate):
        """Compute the discriminating power of a candidate pair.

        Input
            candidates : index of the two candidates concerned

        In this method we compute the deleted points supposing that candidate0
        will be preffered over candidate1 (del0).

        The points deleted in the other case (del1) are simply all_pts - del0
        """
        c = candidate

        # Number of points rejected if candidate[0] preferred over candidate[1]
        del0 = 0

        ties = 0

        for pt in self.admissible_points:
            scoreA = self.get_score(self.alternatives[c[0]], pt)
            scoreB = self.get_score(self.alternatives[c[1]], pt)
            if scoreA < scoreB:
                del0 += 1
            elif scoreA == scoreB:
                ties += 1
        if (self.PII_ranking.index(c[0]) > self.PII_ranking.index(c[1])):
            del0 += len(self.correct_points)

        del1 = (len(self.admissible_points) + len(self.correct_points)
                - del0 - ties)

        return min(del0, del1)

    def is_admissible_constraint(self, indices):
        """Check if this set of alternatives has already been considered."""
        if indices[0] == indices[1]:
            return False
        for const in self.constraints:
            if indices[0] in const and indices[1] in const:
                return False
        return True

    def get_score(self, alt, point):
        """Compute the score of alt with the RS represented by point.

        This is not the real refferenced score. This should be divided by the
        number of referencs but it is useless in our application.
        """
        score = 0
        for i in range(self.crit_number):
            for r in range(self.ref_number):
                pref_function = self.promethee.pref_functions[i]
                diff = alt[i] - point[r][i]
                pref = pref_function.value(diff) - pref_function.value(-diff)
                score += self.promethee.weights[i]*pref
        return score

    def ask_question(self, query):
        """Ask a query to DM and add appropriate constraint to the constraints.

        The query has the form (alternatives)
        """
        r_alt0 = self.PII_ranking.index(query[0])
        r_alt1 = self.PII_ranking.index(query[1])
        if (r_alt0 < r_alt1):
            alternatives = (query[0], query[1])
        else:
            alternatives = (query[1], query[0])

        self.add_constraint(alternatives)

    def add_constraint(self, constraint):
        """Add a constraint to the list of constraints.

        This contstaint must have the form (type, (alernatives) with
            type = 0 : for a PAC constraint
                   1 : for a POS constraint
                   2 : for a ASR constraint
            alternatives : already ordered/ranked alternatives

        For the moment only PAC constraints are handled.
        """
        self.constraints.append(constraint)
        self.update_points(constraint)
        # self.add_random_points()

    def update_points(self, constraint):
        """Check which points of the hypervolume are still admissible.

        This function should be used to check if the existing points do satisfy
        the last constraint added only.
        """
        c = constraint
        if len(self.admissible_points) == 0:
            return
        still_valid = []
        for pt in self.admissible_points:
            scoreA = self.get_score(self.alternatives[c[0]], pt)
            scoreB = self.get_score(self.alternatives[c[1]], pt)
            if scoreA >= scoreB:
                still_valid.append(pt)

        self.admissible_points = still_valid

    def check_for_errors(self):
        """Check all points for a possible mistake."""
        for pt in self.admissible_points:
            refflows = self.referenced.compute_scores(ref_set=pt)
            ranking = self.referenced.compute_ranking(refflows)
            if not self.is_admissible(ranking):
                print("There was an error in the procudure, you are doomed")
                return False
        return True

    def round_analyse(self):
        """Analyse the admissible but not correct points.

        Return the best and worst kendall tau of the admissible points and
        the number of possible rankings.
        """
        all_taus = []
        min_kendall = 1
        max_kendall = -1

        rankings = dict()
        for pt in self.admissible_points:
            RS = pt
            refflows = self.referenced.compute_scores(ref_set=pt)
            ref_ranking = self.referenced.compute_ranking(refflows)

            tau = stats.kendalltau(ref_ranking, self.PII_ranking)[0]
            all_taus.append(tau)
            if tau < min_kendall:
                min_kendall = tau
            if tau > max_kendall:
                max_kendall = tau

            rank = tuple(ref_ranking)
            rankings[rank] = rankings.get(rank, 0) + 1

        for i in range(len(self.correct_points)):
            max_kendall = 1
            all_taus.append(1)

        self.kendall_taus.append(all_taus)
        if rankings:
            self.commonest_ranking = max(rankings, key=rankings.get)
            # prevent the randomness if equality
            number = rankings.get(self.commonest_ranking)
            for key in rankings:
                if key < self.commonest_ranking:
                    self.commonest_ranking = key

        mean_kendall = np.mean(all_taus)
        median_kendall = np.median(all_taus)

        # Rounding up the extremas:
        min_kendall = int(min_kendall*1000)/1000
        max_kendall = int(max_kendall*1000)/1000
        mean_kendall = int(mean_kendall*1000)/1000
        med_kend = int(median_kendall*1000)/1000

        return len(rankings), min_kendall, max_kendall, mean_kendall, med_kend

    def round_query_dm(self):
        """Ask the appropriate question to DM and add constraint."""
        candidates = self.query_cadidates()
        best_cand = self.select_best_candidate(candidates)
        self.ask_question(best_cand)

    def round_add_points(self):
        """Add points until self.desired_points our maxit iterations."""
        tot_pts = len(self.admissible_points) + len(self.correct_points)
        maxit = 5
        while (tot_pts < ((self.desired_points))*(2/3) and maxit >= 0):
            self.divide_points()
            tot_pts = len(self.admissible_points) + len(self.correct_points)
            maxit -= 1
        maxit = 1500
        while (tot_pts < self.desired_points and maxit > 0):
            self.add_random_points()
            tot_pts = len(self.admissible_points) + len(self.correct_points)
            maxit -= 1
        if(tot_pts == 0):
            print('No points can be found anymore ...')
            exit()

    def execute(self, max_rounds=20):
        """Execute one round of the procudure."""
        print("it |init pts | corrects | ranks "
              + "|tau_min|tau_max|tau_mu |tau_med| center ok| deleted ")

        for round in range(max_rounds - 1):
            self.iteration += 1
            corr_pts = len(self.correct_points)
            old_tot_pts = len(self.admissible_points) + corr_pts

            self.rankings, tau_min, tau_max, tau_mean, tau_median = \
                self.round_analyse()

            # check the center
            # flows_center = self.referenced.compute_scores(ref_set=self.center)
            # ranking_center = self.referenced.compute_ranking(flows_center)
            # center_ok = str(self.is_admissible(ranking_center))
            center_ok = " "

            self.round_query_dm()
            new_tot_pts = len(self.admissible_points) + corr_pts
            pts_deleted = old_tot_pts - new_tot_pts

            print(self.it_template.format(self.iteration, old_tot_pts, corr_pts,
                                          self.rankings, tau_min, tau_max,
                                          tau_mean, tau_median, center_ok,
                                          pts_deleted))
            self.round_add_points()

        self.iteration += 1
        corr_pts = len(self.correct_points)
        old_tot_pts = len(self.admissible_points) + corr_pts
        # check the center
        # flows_center = self.referenced.compute_scores(ref_set=self.center)
        # ranking_center = self.referenced.compute_ranking(flows_center)
        #  center_ok = str(self.is_admissible(ranking_center))
        center_ok = " "
        self.rankings, tau_min, tau_max, tau_mean, tau_median = \
            self.round_analyse()

        pts_deleted = 0
        print(self.it_template.format(self.iteration, old_tot_pts, corr_pts,
                                      self.rankings, tau_min, tau_max,
                                      tau_mean, tau_median,
                                      center_ok, pts_deleted))
        self.check_for_errors()
        return self.correct_points
