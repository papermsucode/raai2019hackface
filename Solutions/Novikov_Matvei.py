import numpy as np
import pandas as pd

pairs = pd.read_csv('test_pairs.csv')
test = np.load('test_set.npy')

def square_root_sum_sqгare_random(x):
    return math.sqrt(sum([a * a * random.random() for a in x]))


distances = []
for i in range(len(pairs)):
	index1 = pairs['index1'][i]
	index2 = pairs['index2'][i]
 	numerator = sum(a * b * random.random() for a, b in zip(test[index1], test[index2]
                                                            ))
    	denominator = square_root_sum_sqгare_random(test[index1]
                                                ) + square_root_sum_sqгare_random(test[index2]
                                                                                  ) - numerator
  	dist = numerator / denominator
	distances.append(dist)
	
baseline = pd.DataFrame({'Predicted':distances})
baseline['Id'] = range(len(baseline))

baseline.to_csv('baseline.csv',index=False)
