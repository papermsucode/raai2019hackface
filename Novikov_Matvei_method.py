import numpy as np
import pandas as pd

pairs = pd.read_csv('test_pairs.csv')
test = np.load('test_set.npy')

distances = []
for i in range(len(pairs)):
	index1 = pairs['index1'][i]
	index2 = pairs['index2'][i]
	dist = np.sum((test[index1]-test[index2])**2)
	distances.append(dist)
	
baseline = pd.DataFrame({'Predicted':distances})
baseline['Id'] = range(len(baseline))

baseline.to_csv('baseline.csv',index=False)
