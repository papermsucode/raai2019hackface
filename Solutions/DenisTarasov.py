# -*- coding: utf-8 -*-
"""FinalSolution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T-IxSxzTP4sWe0XInosnxpHZhlw721mD
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, braycurtis, yule, rogerstanimoto

def my_dist(u, v):
  return cosine(u, v) * yule(u, v) * braycurtis(u, v) * np.abs(rogerstanimoto(u, v))

pairs = pd.read_csv('test_pairs.csv') 
test = np.load('test_set.npy') 
distances = [] 

for i in range(len(pairs)): 	
  index1 = pairs['index1'][i] 	
  index2 = pairs['index2'][i] 	
  dist = my_dist(test[index1], test[index2])
  distances.append(dist) 	 
baseline = pd.DataFrame({'Predicted':distances}) 
baseline['Id'] = range(len(baseline)) 
baseline.to_csv('Answers.csv',index=False)