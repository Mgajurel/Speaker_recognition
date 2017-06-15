#PCA-Principal Component Analysis
#To reduce the dimensions of the feature vectors obtained from MFCC
#Author: Surendra Shrestha
#Date: 2017, June-13
#GitHub username: suren37

import numpy as np

class PCA(object):
    def __init__(self):
        pass

    def pca(self, data):
		centered_data = data - np.mean(data)
		U, S, V = np.linalg.svd(centered_data, full_matrices=False)
		components = V
		coefficients = np.dot(U, np.diag(S))
		return coefficients
	
