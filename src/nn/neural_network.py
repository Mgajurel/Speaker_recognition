from scipy.io import wavfile
import os
import numpy as np
import pickle
import alsaaudio, wave

if __name__ == '__main__':
    import sys
    sys.path.append("..")

from feature.MFCC import mfcc
from feature.MFCC import delta

from feature.sigproc import remove_silence
from feature.sigproc import silence_zone

debug = True

def print_label(text, character="*"):
    star = int((80-len(text))/2)
    print(character*star, text, character*star)

def dprint(message):
    if debug:
        print(message)

class NeuralNetwork:
    def __init__(self, filepath="files", is_delta_mode=False, verbose=False):
        self.verbose = verbose
        self.message = ""
        self.filepath = filepath
        self.is_delta = is_delta_mode

        # Load files
        try:
            self.NN = pickle.load(open(self.filepath+'/model.pkl','rb'))

            # Load user names
            userList = open(self.filepath+"/metadata.txt", "r")
            self.users = userList.readlines()
            userList.close()
        except:
            print("Model and metadata.txt not found.")

        if self.verbose:
            print("Delta Mode enable = ", is_delta_mode)

    def train(self):
        self.message = "Training result:"
        # File path
        wavpath=self.filepath+"/wav"
        if self.verbose:
            print("Wav files from: " + wavpath)

        # List all the filenames of wav directory
        wav_files = [f for f in os.listdir(wavpath) if f.endswith('.wav')]
        n = len(wav_files)
        if(n < 1):
            assert("No wav files found at " + wavpath + ". Cancelling operation")
            return

        features = None
        target = None
        is_firstrun = True
        row = 0
        col = 0
        userList = open(self.filepath+"/metadata.txt", "w")

        # Loop until all files are converted to csv
        for audio in wav_files:
            fs, signal = wavfile.read(wavpath+"/"+audio)
            # remove silence
            signal = remove_silence(fs, signal)
            # Extract features from audio signal
            mfcc_feat = mfcc(signal, fs)
            if self.is_delta:
                mfcc_feat = delta(mfcc_feat, 2)

            # save feature as csv files for debugging
            np.savetxt(self.filepath+"/csv/"+audio[7:-4]+".csv", mfcc_feat, fmt='%.8f', delimiter=',')

            # save the username to database
            userList.write(audio[7:-4]+"\n")
            if self.verbose:
                print("\nFile:", audio)
                print("Feature: ", mfcc_feat.shape)

            if is_firstrun:
                features = mfcc_feat
                target = np.zeros(shape= (mfcc_feat.shape[0],n), dtype = int)
                for i in range(mfcc_feat.shape[0]):
                    target[row, col] = 1
                    row += 1
                col += 1
                is_firstrun = False
            else:
                features = np.vstack((features, mfcc_feat))
                for i in range(mfcc_feat.shape[0]):
                    target_row = np.zeros(shape= (n), dtype = int)
                    target_row[col] = 1
                    target = np.vstack((target, target_row))
                    row += 1
                col += 1

        userList.write("Anonymous\n")
        userList.close()

        # Load user names
        userList = open(self.filepath+"/metadata.txt", "r")
        self.users = userList.readlines()
        userList.close()

        features = features.astype(np.float)

        if self.verbose:
            print_label("Data sets", character="-")
            print("Features:", features.shape)
            print(features)
            print("\nTarget:", target.shape)
            print(target)

            print_label("Neural Network modelling", character="-")
            print("Modelling started, this may take a while")

        # Features and target are made now train them

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

        mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30),
            max_iter=10000, tol=1e-6)

        mlp.fit(X_train,y_train)
        self.message += "\nTotal iteration run: %d" %mlp.n_iter_

        predictions = mlp.predict(X_test)

        # Display confusion matrix
        correct = 0
        incorrect = 0
        for i in range(y_test.shape[0]):
            if(y_test[i] == predictions[i]).all():
                correct += 1
            else:
                incorrect += 1
        accuracy = correct / predictions.shape[0] * 100
        self.message += "\nAccurate = %d \nIncorrect = %d \nAccuracy = %.2f%%" %(correct, incorrect, round(accuracy, 2))

        # Training is now complete, save the model
        pickle.dump(mlp, open(self.filepath+'/model.pkl','wb'))

        self.message += "\nModel is saved to %s/model.pkl" %self.filepath

        if self.verbose:
            print(self.message)

        # Load files
        self.NN = pickle.load(open(self.filepath+'/model.pkl','rb'))

        # Load user names
        userList = open(self.filepath+"/metadata.txt", "r")
        self.users = userList.readlines()
        userList.close()

        return self.message

    # predict the output from given test audio files
    def test_predict(self):
        self.message = "Prediction result:"
        # Load files
        self.NN = pickle.load(open(self.filepath+'/model.pkl','rb'))

        # Load user names
        userList = open(self.filepath+"/metadata.txt", "r")
        self.users = userList.readlines()
        userList.close()

        if self.verbose:
            print("Users are ", self.users)

        try:
            wav_path = self.filepath+"/test/wav"
            wav_files = [f for f in os.listdir(wav_path) if f.endswith('.wav')]
            if(len(wav_files) < 1):
                assert("No test wav files found at %s, Cancel operation" %wav_path)
                return

            self.message += "\nTest file path: %s/test" %wav_path

            for wav_file in wav_files:
                self.message += "\nGiven wav file: %s" %wav_file
                fs, signal = wavfile.read(wav_path+"/"+ wav_file)
                # remove silence
                signal = remove_silence(fs, signal)
                # Extract feature
                mfcc_feat = mfcc(signal, fs)
                if self.is_delta:
                    mfcc_feat = delta(mfcc_feat, 2)
                # Compute output from feature
                output = self.NN.predict(mfcc_feat)
                name = self.get_label(output)
                self.message += "\nThe user is %s" %name
        except (FileNotFoundError, IOError):
                print("Wrong file or file path")

        if self.verbose:
            print(self.message)

        return self.message
    # Real time prediction
    def prediction(self, noise_level=2000000):
        #Record wav file ~0.5 sec
        record_wav("test.wav")

        fs, signal = wavfile.read("test.wav")
        signal = remove_silence(fs, signal)

        if(silence_zone(signal, noise_level)):
            return "Silent"
        # Extract feature from the file
        mfcc_feat = mfcc(signal, fs)
        if self.is_delta:
            mfcc_feat = delta(mfcc_feat, 2)

        output = self.NN.predict(mfcc_feat)
        username = self.get_label(output)

        return username
    # Get a label from the given output of prediction
    def get_label(self, output):
        n = output.shape[1]
        count_array = np.zeros((n+1,), dtype=np.int)
        test_data = np.zeros((n,), dtype=np.int)

        for i in range(output.shape[0]):
            data = output[i]
            counted = False
            for j in range(n):
                test_data = np.zeros((n,), dtype=np.int)
                test_data[j] = 1
                if(data == test_data).all():
                    count_array[j] += 1
                    counted = True
            if counted == False:
                count_array[n] += 1

        accuracy = np.amax(count_array) / output.shape[0] * 100

        if(count_array.argmax() < len(self.users)):
            if accuracy > 20:
                label = self.users[count_array.argmax()]
            else:
                label = self.users[n]

        if self.verbose:
            self.message += "\nPrediction Outcome"
            self.message += "\nCount: %s" %np.array_str(count_array)
            self.message += "\nAccuracy = %.2f%%" %round(accuracy, 2)
            self.message += "\nTotal count = %d" %np.amax(count_array)

        return label

    def set_delta(self, delta):
        self.is_delta = delta

    def set_verbose(self, verbose):
        self.verbose = verbose

def record_wav(filename, time=7):
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    inp.setchannels(1)
    inp.setrate(8000)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(1024)
    w = wave.open(filename, 'w')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(8000)

    for i in range(time): #~0.5 seconds
        l, data = inp.read()
        w.writeframes(data)

    w.close()

if __name__ == '__main__':
    nn = NeuralNetwork(filepath="../files")
    print_label("Training")
    print(nn.train())
    print_label("Testing from file...")
    print(nn.test_predict())
