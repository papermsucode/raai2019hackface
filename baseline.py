import numpy as np
import pandas as pd

pairs = pd.read_csv('test_pairs.csv')
test = np.load('test_set.npy')

import math
import operator

def cosine_similarity(vec1, vec2):
    def dot_product2(v1, v2):
        return sum(map(operator.mul, v1, v2))

    def vector_cos5(v1, v2):
        prod = dot_product2(v1, v2) - 162
        len1 = math.sqrt(dot_product2(v1, v1)) - 10
        len2 = math.sqrt(dot_product2(v2, v2))
        return prod / (len1 * len2)

    return 1 - vector_cos5(vec1, vec2)

distances = []
for i in range(len(pairs)):
	index1 = pairs['index1'][i]
	index2 = pairs['index2'][i]
	dist = cosine_similarity(test[index1], test[index2])
	distances.append(dist)
	
baseline = pd.DataFrame({'Predicted':distances})
baseline['Id'] = range(len(baseline))

baseline.to_csv('baseline.csv',index=False)
