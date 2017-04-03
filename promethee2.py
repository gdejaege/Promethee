#!/usr/bin/env python3
"""Promethee and other MCDA method implementation.

The file is subdivided as follow:
    - preferences functions classes. Classical, then symetric versions
    - utility functions : bunch of functions that helps for various tasks
    - mcda methods : netflow computations, preference matrix computation, etc.
"""
from math import exp
import csv
import random
import data_reader as dr


"""Preference function classes."""


class PreferenceType2:

    """Quasi-criterion."""

    q = 0

    def __init__(self, valQ):
        """Constructor."""
        self.q = valQ

    def value(self, diff):
        """Value."""
        if (diff <= self.q):
            return 0
        return 1


class PreferenceType5:

    """Linear preferences and indifference zone."""

    q = 0
    p = 1

    def __init__(self, valQ, valP):
        """Constructor."""
        self.q = valQ
        self.p = valP

    def value(self, diff):
        """Value."""
        if (diff <= self.q):
            return 0
        if (diff <= self.p):
            return (diff - self.q) / (self.p - self.q)
        return 1


class PreferenceType6:

    """Gaussian criterion."""

    s = 0.5
    valSquare = 0.5

    def __init__(self, valS):
        """Constructor."""
        self.s = valS
        self.valSquare = -1 * (2 * valS * valS)

    def value(self, diff):
        """Value."""
        if (diff <= 0):
            return 0
        return 1 - exp(diff * diff / self.valSquare)


class GeneralizedType5:

    """Symetric type5 criterion."""

    q = 0
    p = 1

    def __init__(self, valQ, valP):
        """Constructor."""
        self.q = valQ
        self.p = valP

    def value(self, diff):
        """Value."""
        if (abs(diff) <= self.q):
            return 0
        res = 0
        if(abs(diff) <= self.p):
            res = (abs(diff) - self.q)/(self.p - self.q)
        else:
            res = 1
        if (diff > 0):
            return res
        else:
            return - res


"""Complementary functions."""


class PrometheeII:

    """PrometheeII class."""

    def __init__(self, alternatives, seed=0, alt_num=-1, ceils=None,
                 weights=None, coefficients=None):
        """Constructor of the PrometheeII class."""
        self.alternatives, self.weights, self.coefficients = \
            self.random_parameters(seed, alternatives, alt_num)

        if(weights is not None):
            self.weights = weights
        if(coefficients is not None):
            self.coefficients = coefficients

        # Preference functions
        diffs = self.max_diff_per_criterion(self.alternatives)
        if (ceils is None):
            ceils = [diffs[i] * coeff
                     for i, coeff in enumerate(self.coefficients)]
        self.pref_functions = [PreferenceType5(0, ceil) for ceil in ceils]

        self.scores = self.compute_scores()
        self.ranking = self.compute_ranking(self.scores)

    def random_parameters(self, seed, alternatives, nmbre=-1):
        """Compute random parameters using a seed."""
        random.seed(seed)
        if nmbre != -1:
            alternatives = random.sample(alternatives, nmbre)

        coefficients = [random.uniform(0.4, 1) for i in alternatives[0]]
        weights = [random.randint(30, 100) for i in alternatives[0]]
        weights = [w/sum(weights) for w in weights]
        # print(weights)
        # print(coefficients)
        # print(alternatives[0])
        return alternatives, weights, coefficients

    def max_diff_per_criterion(self, alternatives, crit_quantity=-1):
        """Retun a list of delta max for each criterion.

        Should not be used sinc we can use sort by criterion.
        """
        # quantity of criteria we are looking at
        if crit_quantity == -1:
            crit_quantity = len(alternatives[0])

        eval_per_criterion = []
        # evaluations of the alternatives for each criterion individually
        for criterion in range(crit_quantity):
            eval_per_criterion.append([row[criterion] for row in alternatives])

        diff = []
        for criterion in eval_per_criterion:
            diff.append(max(criterion) - min(criterion))

        return diff

    def compute_pairwise_pref(self, alternatives, weights, funcPrefCrit):
        """Return the pairwise preference matrix Pi."""
        pi = [[0 for alt in alternatives] for alt in alternatives]
        for i, alti in enumerate(alternatives):
            for j, altj in enumerate(alternatives):
                for k in range(len(weights)):
                    weight = weights[k]
                    funcPref = funcPrefCrit[k]
                    valAlti = alti[k]
                    valAltj = altj[k]
                    diff = valAlti - valAltj
                    # val2 = valAlt2 - valAlt1
                    pi[i][j] += weight * funcPref.value(diff)
                    pi[j][i] += weight * funcPref.value(-diff)
        return pi

    def compute_scores(self, alternatives=None, weights=None, pref_funcs=None):
        """Compute the score of this of the alternatives with this method.

        Please redefine this function for child classes.
        """
        if alternatives is None:
            alternatives = self.alternatives
        if weights is None:
            weights = self.weights
        if pref_funcs is None:
            pref_funcs = self.pref_functions

        return self.compute_netflow(alternatives, weights, pref_funcs)

    def compute_netflow(self, alternatives, weights, pref_func):
        """Compute the netflows of the alternatives.

        Inuput :
            alternatives : matrice
            weights : list
            funcPrefCrit : list of functions
        Output :
            netflows[i] = score of the ith alternative in alternatives(input)
        """
        netflows = []
        if (len(alternatives) == 1):
                return [1]

        for ai in alternatives:
            flow = 0
            for aj in alternatives:
                if ai == aj:
                    continue
                for k in range(len(weights)):
                    weight = weights[k]
                    pref_k = pref_func[k]
                    ai_k = ai[k]
                    aj_k = aj[k]
                    diff = ai_k - aj_k
                    flow += weight*(pref_k.value(diff) - pref_k.value(-diff))
                    Pi_ij = Pi_ij + weight*pref_func_k.value(diff)
                    Pi_ji = Pi_ji + weight*pref_func_k.value(-diff)
                posflow = posflow + Pi_ij
                negflow = negflow + Pi_ji
            posflow = posflow / (len(alternatives) - 1)
            negflow = negflow / (len(alternatives) - 1)
            netflow = posflow - negflow
            netflows.append(netflow)
        return netflows

    def compute_ranking(self, scores):
        """Return the ranking given the netflows.

        ranking[i] = index in the alternatives (input) of the alternative
                    which is ranked ith.
        scores[i] = score of the ith alternative in of the alternatives(input)
        """
        # sort k=range(...) in decreasing order of the netflows[k]
        ranking = sorted(range(len(scores)), key=lambda k: scores[k],
                         reverse=True)
        return ranking

    def compute_rr_number(self, verbose=False):
        """Compute the total number of rank reversals on the method.

        verbose input only serves for printing debug information
        """
        RR = 0
        for i in range(len(self.alternatives)):
            copy_alternatives = self.alternatives[:]
            del copy_alternatives[i]
            scores = self.compute_scores(alternatives=copy_alternatives)
            # print(scores)
            ranks = self.compute_ranking(scores)
            """Since we applied the method on n-1 alternatives, the ranks
            will be in [0, n-1] instead of [0, n]"""
            for j in range(len(ranks)):
                if ranks[j] >= i:
                    ranks[j] += 1
            # print(ranks)
            RR += self.compare_rankings(self.ranking, ranks, i, verbose)
        return RR

    def compare_rankings(self, init_ranking, new_ranking, deleted_alt,
                         verbose=False):
        """Compute the number of rank reversal between two rankings.

        Input :
            init_ranking : ranking obtained with all alternatives
            new_ranking : ranking obtained when deleted_alt is removed from
                          the set of alternatives.
        """
        init_copy = init_ranking[:]
        new_copy = new_ranking[:]
        init_copy.remove(deleted_alt)

        RR = 0
        while(len(init_copy) > 0):
            j = 0
            while (init_copy[0] != new_copy[j]):
                if (verbose):
                    print("RR between " + str(init_copy[0]) + " and "
                          + str(new_copy[j]) + " when " + str(deleted_alt)
                          + "is deleted")
                RR += 1
                j += 1
            del init_copy[0]
            del new_copy[j]
        return RR


class RobustPII(PrometheeII):

    """Robust PII class."""

    def __init__(self, alternatives, seed=0, alt_num=-1, coefficients=None,
                 weights=None, pref_func=None, ceils=None, R=1000, m=5):
        """Constructor."""
        self.R = R
        self.m = m
        super().__init__(alternatives=alternatives, seed=seed, alt_num=alt_num,
                         ceils=ceils, coefficients=coefficients,
                         weights=weights)
        if(m > len(self.alternatives)):
            exit("lol m>alternatives")

    def compute_scores(self, alternatives=None, weights=None, pref_funcs=None,
                       R=None, m=None):
        """Compute the robust promethee score."""
        if alternatives is None:
            alternatives = self.alternatives
        if weights is None:
            weights = self.weights
        if pref_funcs is None:
            pref_funcs = self.pref_functions
        if R is None:
            R = self.R
        if m is None:
            m = self.m

        return self.compute_robflow(alternatives, weights, pref_funcs, R, m)

    def compute_robflow(self, alternatives, weights, pref_funcs, R, m):
        """Return the robust flow."""
        random.seed()
        Pij = [[0 for i in range(len(alternatives))]
               for j in range(len(alternatives))]
        for iteration in range(R):
            indices = random.sample(range(len(alternatives)), m)
            Am = [alternatives[i] for i in indices]
            score = self.compute_netflow(Am, weights, pref_funcs)
            ranking_0_to_m = self.compute_ranking(score)
            ranking = [indices[i] for i in ranking_0_to_m]
            for i, ai in enumerate(ranking[:-1]):
                for aj in ranking[i+1:]:
                    Pij[ai][aj] += 1

        # Normalization
        for i in range(1, len(Pij)):
            for j in range(i):
                tot = Pij[i][j] + Pij[j][i]
                if(tot != 0):
                    Pij[i][j] = Pij[i][j]/tot
                    Pij[j][i] = Pij[j][i]/tot

        # Netflow of a matrix
        rob_flows = []
        for i in range(len(Pij)):
            sum_row_i = sum([Pij[i][j] for j in range(len(Pij))])
            rob_flows.append((1/(len(Pij)-1))*(2*sum_row_i) - 1)
        return rob_flows
