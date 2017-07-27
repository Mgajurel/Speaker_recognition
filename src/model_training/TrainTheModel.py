import csv
import numpy as np
from sknn.mlp import Classifier, Layer
import pickle
import os

# Functions to extract feature/save values from csv and return the value
def csv_extractor(csv_file):
    with open(csv_file,'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter = ',')
        data = [data for data in data_iter]
    feat = np.asarray(data, dtype = None)
    feat = feat.astype(np.float)
    return feat

def print_label(text, character="*"):
    star = int((80-len(text))/2)
    print(character*star, text, character*star)

def training(filepath):
    print_label("Training the dataset")

    csv_files = [f for f in os.listdir(filepath) if f.endswith('.csv')]
    n = len(csv_files)
    if(n < 1):
        print("No csv files found. Cancelling operation.")
        return
    print("Total number of Class =",n)

    sample_size = 1200
    features = None
    is_firstrun = True

    row = 0
    col = 0
    target = None

    target = np.zeros(shape= (sample_size*n,n), dtype = int)
    userList = open("files/metadata.txt", "w")
    for csv in csv_files:
        print("CSV file:", csv)
        feature = csv_extractor(filepath + "/" + csv)
        print("Size of feature full", feature.shape)       
        feature = feature[0:sample_size]
        print("Size of feature used", feature.shape, "\n")
        userList.write(csv[7:-9] + "\n") 

        # # Increase row iteratively but increase column only after end of 1 feature
        # for i in range(sample_size):
        #     target[row, col] = 1
        #     row += 1
        # col += 1

        if is_firstrun:
            features = feature
            target = np.zeros(shape= (feature.shape[0],n), dtype = int)
            for i in range(feature.shape[0]):
                target[row, col] = 1
                row += 1
            col += 1
            is_firstrun = False
        else:
            features = np.vstack((features, feature))
            for i in range(feature.shape[0]):
                target_row = np.zeros(shape= (n), dtype = int)
                target_row[col] = 1
                target = np.vstack((target, target_row))
                row += 1
            col += 1

    userList.close()

    print_label("Data sets", character="-")
    features = features.astype(np.float)
    print("Features:", features.shape)
    print(features)
    print("Target:", target.shape)
    print(target)

    print_label("Neural Network modelling", character="-")
    print("Modelling started, this may take a while")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, target)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))

    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import MinMaxScaler
    # mlp = Pipeline([
    #     ('min/max scaler', MinMaxScaler(feature_range=(-40.0, 40.0))),
    #     ('neural network', Classifier(layers =[Layer("Sigmoid", units = 25), Layer("Sigmoid")],
    #                     learning_rate = 0.001,
    #                     n_iter = 2))])

    mlp.fit(X_train,y_train) 

    predictions = mlp.predict(X_test)   

    # Display confusion matrix
    confusion_matrix(y_test, predictions)

    print("Neural network training finish")

    # Training is now complete, save the model
    pickle.dump(mlp, open(filepath+'/../model.pkl','wb'))
    print("Model is saved to files/model.pkl")

def prediction(feature_ndarray):
    print("Loading database file")
    NN = pickle.load(open('files/model.pkl','rb'))

    # Load user names
    userList = open("files/metadata.txt", "r")
    users = userList.readlines()
    userList.close()

    print("Predecting from features")
    #Normalize the test data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(feature_ndarray)

    # Now apply the transformations to the data:
    X_test = scaler.transform(feature_ndarray)
    output = NN.predict(X_test)
    
    n = output.shape[1]
    count_array = np.zeros((n,), dtype=np.int)
    count_other = 0
    test_data = np.zeros((n,), dtype=np.int)
    counted = False

    for i in range(output.shape[0]):
        data = output[i]
        for j in range(n):
            test_data = np.zeros((n,), dtype=np.int)
            test_data[j] = 1
            if(data == test_data).all():
                count_array[j] += 1
                counted = True
        if not counted:
            count_other += 1


    # np.set_printoptions(threshold=np.nan)
    print("Predicted Complete:")
    print("Prediction Count", count_array, "Other:", count_other)
    print("Recognized user:", users[count_array.argmax()])

def confusion_matrix(y_test, predictions):
    correct = 0
    incorrect = 0
    for i in range(y_test.shape[0]):
        if(y_test[i] == predictions[i]).all():
            correct += 1
        else:
            incorrect += 1
    print("Accurate =", correct)
    print("Incorrect =", incorrect)

debug = False

def dprint(message):
    if(debug):
        print(message)

class Prediction(object):
    def __init__(self):
        dprint("Load pickle file")
        self.NN = pickle.load(open('files/model.pkl','rb'))
        dprint("Load users from metadata")
        # Load user names
        userList = open("files/metadata.txt", "r")
        self.users = userList.readlines()
        userList.close()
        dprint("Init finished.")

    def predict(self, mfcc_signal):
        dprint("Predicting")

        #Normalize the test data        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(mfcc_signal)

        # Now apply the transformations to the data:
        X_test = scaler.transform(mfcc_signal)
        output = self.NN.predict(X_test)
        print("The user is:", self.get_label(output))
        return output

    def get_label(self, output):
        n = output.shape[1]
        count_array = np.zeros((n,), dtype=np.int)
        count_other = 0
        test_data = np.zeros((n,), dtype=np.int)
        counted = False

        for i in range(output.shape[0]):
            data = output[i]
            for j in range(n):
                test_data = np.zeros((n,), dtype=np.int)
                test_data[j] = 1
                if(data == test_data).all():
                    count_array[j] += 1
                    counted = True
            if not counted:
                count_other += 1


        # np.set_printoptions(threshold=np.nan)
        dprint("Predicted Overview:")
        if debug:
            print("Prediction Count", count_array, "Other:", count_other)
        label = self.users[count_array.argmax()]
        if debug:
            print("Recognized user:", label)
        return label
