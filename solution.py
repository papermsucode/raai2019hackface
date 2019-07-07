import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

pairs = pd.read_csv('test_pairs.csv')
test = np.load('test_set.npy')

fa = FactorAnalyzer(rotation='varimax', n_factors=512)
fa.fit(test)
test_factor = fa.transform(test)

distances = []

#from https://github.com/marcelcaraciolo/crab/blob/master/crab/similarities/similarity_distance.py
def sim_pearson(vector1, vector2, **args):
    '''
    This correlation implementation is equivalent to the cosine similarity
    since the data it receives is assumed to be centered -- mean is 0. The
    correlation may be interpreted as the cosine of the angle between the two
    vectors defined by the users' preference values.
    Parameters:
        vector1: The vector you want to compare
        vector2: The second vector you want to compare
        args: optional arguments
    The value returned is in [0,1].
    '''
    # Using Content Mode.
    if type(vector1) == type({}):
        sim = {}
        [sim.update({item:1})  for item in vector1  if item in vector2]
        n = len(sim)

        if n == 0:
            return 0.0

        sum1 = sum([vector1[it]  for it in sim])
        sum2 = sum([vector2[it]  for it in sim])

        sum1Sq = sum([pow(vector1[it], 2.0) for it in sim])
        sum2Sq = sum([pow(vector2[it], 2.0) for it in sim])

        pSum = sum(vector1[it] * vector2[it] for it in sim)

        num = pSum - (sum1 * sum2 / float(n))

        den = sqrt((sum1Sq - pow(sum1, 2.0) / n) *
                   (sum2Sq - pow(sum2, 2.0) / n))

        if den == 0.0:
            return 0.0

        return num / den
    else:
        # Using Value Mode.
        if len(vector1) != len(vector2):
            raise ValueError('Dimmensions vector1 != Dimmensions vector2')

        if len(vector1) == 0 or len(vector2) == 0:
            return 0.0

        sum1 = sum(vector1)
        sum2 = sum(vector2)

        sum1q = sum([pow(v, 2) for v in vector1])
        sum2q = sum([pow(v, 2) for v in vector2])

        pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])

        num = pSum - (sum1 * sum2 / len(vector1))

        den = sqrt((sum1q - pow(sum1, 2) / len(vector1)) *
                   (sum2q - pow(sum2, 2) / len(vector1)))

        if den == 0.0:
            return 0.0

        return num / den

for i in range(len(pairs)):

    index1 = pairs['index1'][i]
    index2 = pairs['index2'][i]
    dist = 1 - np.absolute(sim_pearson(test_factor[index1], test_factor[index2]))
    distances.append(dist)

baseline = pd.DataFrame({'Predicted':distances})
baseline['Id'] = range(len(baseline))

baseline.to_csv('baseline_17_factor_1_512_cosine_varimax.csv',index=False)
