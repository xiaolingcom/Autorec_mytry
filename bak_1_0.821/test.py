import pickle
import numpy as np
import torch as t

with open('sparseMat_0.9_cv.csv', 'rb') as fs:
    data = pickle.load(fs)

# with open('sparsePieces_cv','rb') as f:
#     data=pickle.load(f)

print("debug")