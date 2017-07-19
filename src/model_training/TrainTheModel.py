import csv
import sys
import numpy as np
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle

def csv_extractor(csv_file):
    with open(csv_file,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter = ',')
        data = [data for data in data_iter]
    feat = np.asarray(data, dtype = None)
    feat = feat.astype(np.float)
    return feat


feat_kushal = csv_extractor('mfcc_kushal.csv')
feat_kushal = feat_kushal[0:772]
print(len(feat_kushal))
feat_laxmi = csv_extractor('mfcc_laxmi.csv')
print(len(feat_laxmi))
feat = np.vstack((feat_kushal,feat_laxmi))
feat = feat.astype(np.float)


y_train = np.zeros(shape= (1544,2), dtype = int)
for i in range(1544):
    if i < 772:
        y_train[i] = [1,0]
    else:
        y_train[i] = [0,1]



#NN modeling

pipeline = Pipeline([
        ('min/max scaler', MinMaxScaler(feature_range=(-40.0, 40.0))),
        ('neural network', Classifier(layers =[Layer("Sigmoid", units = 25), Layer("Sigmoid")],
                        learning_rate = 0.0001,
                        n_iter = 200 ))])

pipeline.fit(feat, y_train)




print("Training Complete and Model is saved")
pickle.dump(pipeline, open('NN.pkl','wb'))
