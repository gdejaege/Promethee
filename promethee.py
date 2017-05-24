#!/usr/bin/env python3
"""Implementation of PrometheeII, Robust Promethee and Referenced Promethee.

The tree methods are implemented in the form of a hierarchy of classes. Some
other usefull functions are implemented such as:
    * classes for the preference functions
    * functions for the strategies that can be used to build the sets of
      reference profiles needed by Referenced Promethee
    * a function checking if the common parameters between two methods
      instanciation are equal.
"""

from math import exp
from scipy import stats
import random
import data_reader as dr


"""Preference function classes."""


class PreferenceType2:

    """Quasi-criterion."""

    q = 0

    def __init__(self, q=0):
        """Constructor."""
        self.q = q

    def value(self, diff):
        """Value."""
        if (diff <= self.q):
            return 0
        return 1


class PreferenceType5:

    """Linear preferences and indifference zone."""

    q = 0
    p = 1

    def __init__(self, q=0, p=1):
        """Constructor."""
        self.q = q
        self.p = p

    def value(self, diff):
        """Value."""
        if (diff <= self.q):
            return 0
        if (diff <= self.p):
            return (diff - self.q) / (self.p - self.q)
        return 1


class GeneralizedType5:

    """Symetric type5 criterion."""

    q = 0
    p = 1

    def __init__(self, q, p):
        """Constructor."""
        self.q = q
        self.p = p

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


def strategy1(alternatives, ref_number=4, seed=0):
    """Build a set of random reference profiles (SRP).

    Inputs:
        alternatives - matrix composed of one list of evaluations for each
                       alternative.
        ref_number - number of profiles composing the set.
        seed - initialisation of the python pseudorandom functions.

    Note: the reference profiles evaluation hower lie between the minimal and
    maximal evaluation of the set of alternatives.
    """
    random.seed(seed)
    SRP = []
    # Transposition of the alternatives: one list of alternatives for each
    # criteria
    eval_per_criterion = list(map(list, zip(*alternatives)))
    for ref in range(ref_number):
        reference = []
        for criterion in eval_per_criterion:
            reference.append(random.uniform(min(criterion), max(criterion)))
        SRP.append(reference)
    return SRP


def strategy2(alternatives, refs_quantity=4, seed=0):
    """Build a set of reference profiles (SRP) as percentiles of the evaluations.

    Inputs:
        alternatives - matrix composed of one list of evaluations for each
                       alternative.
        ref_number - number of profiles composing the set.
        seed - not used but kept to keep the same signature between all
               strategies.
    """
    SRP = []
    eval_per_criterion = list(map(list, zip(*alternatives)))

    for percentile in range(refs_quantity):
        ref = []
        percent = (percentile/(refs_quantity-1))*100
        for criterion in eval_per_criterion:
            ref.append(stats.scoreatpercentile(criterion, percent))
        SRP.append(ref)

    return SRP


def strategy3(alternatives, refs_quantity=4, seed=0):
    """Build a set of ref profiles (SRP) equally spaced between the alternatives.

    Inputs:
        alternatives - matrix composed of one list of evaluations for each
                       alternative.
        ref_number - number of profiles composing the set.
        seed - not used but kept to keep the same signature between all
               strategies.
    """
    SRP = []
    # Transposition of the alternatives: one list of alternatives for each
    # criteria
    eval_per_criterion = list(map(list, zip(*alternatives)))

    min_per_criterion = []
    diff = []
    for criterion in eval_per_criterion:
        diff.append(max(criterion) - min(criterion))
        min_per_criterion.append(min(criterion))

    for i in range(refs_quantity):
        ref = []
        prop = (i/(refs_quantity-1))
        for criterion in range(len(diff)):
            ref.append(diff[criterion]*prop + min_per_criterion[criterion])
        SRP.append(ref)
    return SRP


def strategy4(alternatives, refs_quantity=4, seed=0):
    """Build (SRP) equally spaced in the interquartile range of evaluations.

    Inputs:
        alternatives - matrix composed of one list of evaluations for each
                       alternative.
        ref_number - number of profiles composing the set.
        seed - not used but kept to keep the same signature between all
               strategies.
    """
    SRP = []
    # Transposition of the alternatives: one list of alternatives for each
    # criteria
    eval_per_criterion = list(map(list, zip(*alternatives)))

    for percentile in range(refs_quantity):
        ref = []
        percent = (percentile/(refs_quantity-1))*50 + 25
        for criterion in eval_per_criterion:
            ref.append(stats.scoreatpercentile(criterion, percent))
        SRP.append(ref)

    return SRP


def check_parameters(method1, method2):
    """Check if all the common parameters between the methods are equal."""
    res = True
    # Alternatives
    A1 = method1.alternatives
    A2 = method2.alternatives
    for i in range(len(A1)):
        if A1[i] != A2[i]:
            res = False

    # Weights
    W1 = method1.weights
    W2 = method2.weights
    if W1 != W2:
        res = False

    # Ceils
    C1 = method1.ceils
    C2 = method2.ceils
    if C1 != C2:
        res = False

    return res


class PrometheeII:

    """PrometheeII class."""

    def __init__(self, alternatives, seed=0, alt_num=-1, ceils=None,
                 weights=None, coefficients=None):
        """Constructor of the PrometheeII class.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            seed - seed provided to python pseudo random number generator. It
                   is used to create some random (w, F) for the method if these
                   are not provided as arguments.
            alt_num - quantity of alternatives from 'alternative' which must be
                      kept.
            ceils - list of the values of the strict preference thresholds for
                    all criteria (p).
            weights - list of the relative importance (or weigths) of all
                      criteria.
            coefficients - if 'ceils' is not provided, some new ceils will be
                           computed as these coefficents time the amplitude
                           between the highest and lowest evaluation of each
                           criterion.
        """
        # first, each Promethee parameter is set at random, this value is then
        # overwritten by the the value of the parameters given as argument
        self.alternatives, self.weights, self.coefficients = \
            self.random_parameters(seed, alternatives, alt_num)

        # Transposition of the alternatives: one list of alternatives for each
        # criteria
        self.eval_per_crit = list(map(list, zip(*self.alternatives)))

        if(weights is not None):
            self.weights = weights
        if(coefficients is not None):
            self.coefficients = coefficients

        # Preference functions
        diffs = self.max_diff_per_criterion(self.alternatives)
        if (ceils is None):
            ceils = [diffs[i] * coeff
                     for i, coeff in enumerate(self.coefficients)]
        self.ceils = ceils
        self.pref_functions = [PreferenceType5(0, ceil) for ceil in ceils]

        self.scores = self.compute_scores()
        self.ranking = self.compute_ranking(self.scores)

    def random_parameters(self, s, alternatives, qty=-1):
        """Compute random subset of alternatives and parameters using a seed.

        Inputs:
            s - seed.
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            qty - quantity of alternatives desired, if 'qty' is equal to -1
                  then the set the whole set of alternatives is returned
                  (in the original ordering!).
        """
        random.seed(s)
        if qty != -1:
            alternatives = random.sample(alternatives, nmbre)

        coefficients = [random.uniform(0.4, 1) for i in alternatives[0]]
        weights = [random.randint(30, 100) for i in alternatives[0]]
        weights = [w/sum(weights) for w in weights]
        return alternatives, weights, coefficients

    def max_diff_per_criterion(self, alternatives, crit_quantity=-1):
        """Retun a list of delta max for each criterion.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            crit_quantity - number of criteria for which the amplitude of the
                            maximal difference must be computed.
        """
        # quantity of criteria we are looking at
        if crit_quantity == -1:
            crit_quantity = len(alternatives[0])

        eval_per_criterion = list(map(list, zip(*alternatives)))

        diff = []
        for criterion in eval_per_criterion:
            diff.append(max(criterion) - min(criterion))

        return diff

    def compute_pairwise_pref(self, alternatives, weights, pref_funct_crit):
        """Return the pairwise preference matrix Pi.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funct_crit - list of the preference functions for all criteria.

        """
        pi = [[0 for alt in alternatives] for alt in alternatives]
        for i, alti in enumerate(alternatives):
            for j, altj in enumerate(alternatives):
                for k in range(len(weights)):
                    weight = weights[k]
                    pref_function = pref_funct_crit[k]
                    valAlti = alti[k]
                    valAltj = altj[k]
                    diff = valAlti - valAltj
                    # val2 = valAlt2 - valAlt1
                    pi[i][j] += weight * pref_function.value(diff)
                    pi[j][i] += weight * pref_function.value(-diff)
        return pi

    def compute_scores(self, alternatives=None, weights=None, pref_funcs=None):
        """Compute the score of this of the alternatives with this method.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_functs - list of the preference functions for all criteria.

        Note : this function should be redefined for child classes.
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

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funct_crit - list of the preference functions for all criteria.

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
            flow = flow / (len(alternatives) - 1)
            netflows.append(flow)
        return netflows

    def compute_ranking(self, scores):
        """Return the ranking given the scores.

        Input:
            scores[i] = score of the ith alternative

        Output:
            ranking[i] = index in 'scores' of the alternative ranked ith.
        """
        # sort k=range(...) in decreasing order of the netflows[k]
        ranking = sorted(range(len(scores)), key=lambda k: scores[k],
                         reverse=True)
        return ranking

    def compute_rr_number(self, verbose=False):
        """Compute the total number of rank reversals of the method.

        Notes:
            * verbose only serves for printing debug information
            * the ranking (self.ranking) must already be computed
        """
        total_rr_quantity = 0
        for i in range(len(self.alternatives)):
            copy_alternatives = self.alternatives[:]
            del copy_alternatives[i]
            scores = self.compute_scores(alternatives=copy_alternatives)
            ranks = self.compute_ranking(scores)
            # Since we applied the method on n-1 alternatives, the ranks
            # will be in [0, n-1] instead of [0, n]
            for j in range(len(ranks)):
                if ranks[j] >= i:
                    ranks[j] += 1
            rr_quantity = self.compare_rankings(self.ranking, ranks, i, verbose)
            total_rr_quantity += rr_quantity
        return total_rr_quantity

    def compare_rankings(self, init_ranking, new_ranking, deleted_alt,
                         verbose=False):
        """Compute the number of rank reveSRPal between two rankings.

        Inputs:
            init_ranking - ranking obtained with all alternatives
            new_ranking - ranking obtained when deleted_alt is removed from
                          the set of alternatives.
            deleted_alt - please guess
            verbose - print debuging messages
        """
        init_copy = init_ranking[:]
        new_copy = new_ranking[:]
        init_copy.remove(deleted_alt)

        rr_quantity = 0
        while(len(init_copy) > 0):
            j = 0
            while (init_copy[0] != new_copy[j]):
                if (verbose):
                    print("RR between " + str(init_copy[0]) + " and "
                          + str(new_copy[j]) + " when " + str(deleted_alt)
                          + "is deleted")

                rr_quantity += 1
                j += 1
            del init_copy[0]
            del new_copy[j]
        return rr_quantity

    def analyse_rr(self, verbose=False):
        """Compute the pair of alternatives for which rr occurs.

        Output:
            all_rr_instance - dictionary containing the the pair of alternatives
                              which have had their rank reversed (key), and the
                              quantity of time this reversal happened (value).
        Note:
            * this function is similar to the one counting the number of rank
              reversals but is reimplemented here for more lisibility.
        """
        all_rr_instances = dict()
        for i in range(len(self.alternatives)):
            copy_alternatives = self.alternatives[:]
            del copy_alternatives[i]
            scores = self.compute_scores(alternatives=copy_alternatives)
            ranks = self.compute_ranking(scores)
            """Since we applied the method on n-1 alternatives, the ranks
            will be in [0, n-1] instead of [0, n]"""
            for j in range(len(ranks)):
                if ranks[j] >= i:
                    ranks[j] += 1

            rr_instances = self.get_rr(self.ranking, ranks, i, verbose)

            for key in rr_instances:
                all_rr_instances[key] = \
                    all_rr_instances.get(key, 0) + rr_instances.get(key)
        return all_rr_instances

    def get_rr(self, init_ranking, new_ranking, deleted_alt, verbose=False):
        """Compute the number of rank reversal between two rankings.

        Inputs:
            init_ranking - ranking obtained with all alternatives
            new_ranking - ranking obtained when deleted_alt is removed from
                          the set of alternatives.
        """
        init_copy = init_ranking[:]
        new_copy = new_ranking[:]
        init_copy.remove(deleted_alt)

        rr_instances = dict()
        while(len(init_copy) > 0):
            j = 0
            while (init_copy[0] != new_copy[j]):
                if (verbose):
                    print("RR between " + str(init_copy[0]) + " and "
                          + str(new_copy[j]) + " when " + str(deleted_alt)
                          + "is deleted")

                # add occurrence to dict of rank reveSRPals
                a = max(new_copy[j], init_copy[0])
                b = min(new_copy[j], init_copy[0])
                rr_instances[(a, b)] = rr_instances.get((a, b), 0) + 1

                j += 1
            del init_copy[0]
            del new_copy[j]
        return rr_instances


class RobustPII(PrometheeII):

    """Robust PII class."""

    def __init__(self, alternatives, seed=0, alt_num=-1, coefficients=None,
                 weights=None, pref_func=None, ceils=None, R=1000, m=5):
        """Constructor.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            seed - seed provided to python pseudo random number generator. It
                   is used to create some random (w, F) for the method if these
                   are not provided as arguments.
            alt_num - quantity of alternatives from 'alternative' which must be
                      kept.
            coefficients - if 'ceils' is not provided, some new ceils will be
                           computed as these coefficents time the amplitude
                           between the highest and lowest evaluation of each
                           criterion.
            weights - list of the relative importance (or weigths) of all
                      criteria.
            pref_func - list of preference functions for all criteria
            ceils - list of the values of the strict preference thresholds for
                    all criteria (p).
            R - number of repetions of the PrometheeII methods on samples of
                the alternatives.
            m - size of these samples.
        """
        self.R = R
        self.m = m
        super().__init__(alternatives=alternatives, seed=seed, alt_num=alt_num,
                         ceils=ceils, coefficients=coefficients,
                         weights=weights)
        if(m > len(self.alternatives)):
            exit("m > alternatives, this is not possible")

    def compute_scores(self, alternatives=None, weights=None, pref_funcs=None,
                       R=None, m=None):
        """Compute the robust promethee score.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funcs - list of the preference functions for all criteria.
            R - number of repetions of the PrometheeII methods on samples of
                the alternatives.
            m - size of these samples.
        """
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
        """Return the robust flow.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funcs - list of the preference functions for all criteria.
            R - number of repetions of the PrometheeII methods on samples of
                the alternatives.
            m - size of these samples.
        """
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


class ReferencedPII(PrometheeII):

    """Referenced Promethee class."""

    def __init__(self, alternatives, seed=0, alt_num=-1, coefficients=None,
                 weights=None, pref_func=None, ceils=None, ref_set=None,
                 strategy=None, ref_num=4):
        """Constructor.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            seed - seed provided to python pseudo random number generator. It
                   is used to create some random (w, F) for the method if these
                   are not provided as arguments.
            alt_num - quantity of alternatives from 'alternative' which must be
                      kept.
            coefficients - if 'ceils' is not provided, some new ceils will be
                           computed as these coefficents time the amplitude
                           between the highest and lowest evaluation of each
                           criterion.
            weights - list of the relative importance (or weigths) of all
                      criteria.
            pref_func - list of preference functions for all criteria
            ceils - list of the values of the strict preference thresholds for
                    all criteria (p).
            ref_set - matrix composed of one list of evaluations for each
                      reference profile.
            strategy - strategy used to build a set of reference profiles.
            ref_num - quantity of reference profiles needed in a set.
        """
        if (ref_set is not None):
            self.SRP = ref_set
        elif (strategy is not None):
            self.SRP = strategy(alternatives, ref_num, seed)
        else:
            print("precise a references set or a strategy to build one")
            exit()

        super().__init__(alternatives=alternatives, seed=seed, alt_num=alt_num,
                         ceils=ceils, coefficients=coefficients,
                         weights=weights)

    def compute_scores(self, alternatives=None, weights=None, pref_funcs=None,
                       ref_set=None):
        """Compute the referenced promethee score.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funcs - list of the preference functions for all criteria.
                the alternatives.
            ref_set - matrix composed of one list of evaluations for each
                      reference profile.
        """
        if alternatives is None:
            alternatives = self.alternatives
        if weights is None:
            weights = self.weights
        if pref_funcs is None:
            pref_funcs = self.pref_functions
        if ref_set is None:
            ref_set = self.SRP

        return self.compute_refflow(alternatives, weights, pref_funcs, ref_set)

    def compute_refflow(self, alternatives, weights, pref_funcs, SRP):
        """Return the referenced flow.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funcs - list of the preference functions for all criteria.
            SRP - matrix composed of one list of evaluations for each
                      reference profile.
        """
        refflows = []
        for alt in alternatives:
            score = 0
            for ref in SRP:
                for k in range(len(weights)):
                    weight = weights[k]
                    pref_k = pref_funcs[k]
                    alt_k = alt[k]
                    ref_k = ref[k]
                    diff = alt_k - ref_k
                    score += weight*(pref_k.value(diff) - pref_k.value(-diff))
            score = score / (len(SRP))
            refflows.append(score)
        return refflows

    def draws_quantity(self, scores, threshold=1e-3, verbose=False):
        """Compute the quantity of ties.

        Alternatives have the same ranking if their score is closer than
        threshold. Each pair of such alternative will consist in a draw.

        """
        ties = 0
        for i in range(len(scores) - 1):
            for j in range(i + 1, len(scores)):
                if abs(scores[i] - scores[j]) < threshold:
                    ties += 1
        return ties
