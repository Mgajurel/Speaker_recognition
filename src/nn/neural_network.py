from scipy.io import wavfile
import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
# import alsaaudio, wave

if __name__ == '__main__':
    import sys
    sys.path.append("..")

from feature.MFCC import mfcc
from feature.MFCC import delta
from nn.recorder import record_to_file

from feature.sigproc import remove_silence
from feature.sigproc import silence_zone

debug = True

def print_label(text, character="*"):
    star = int((80-len(text))/2)
    print(character*star, text, character*star)

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
            self.users = userList.read().split('\n')
            userList.close()
        except FileNotFoundError:
            print("Model and metadata.txt not found.")

        self.mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation = 'logistic')

        if self.verbose:
            print("Delta Mode enable = ", is_delta_mode)
    # Train the network and generate model.pkl file and csv file
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
        wav_dic = {}

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

            # If user is not already in the list/dictionary add them
            try:
                wav_dic[audio[0:-6]]
            except:
                userList.write(audio[0:-6]+"\n")
                wav_dic[audio[0:-6]] = col
                col += 1
            if self.verbose:
                print("\nFile:", audio)
                print("Feature: ", mfcc_feat.shape)

            if is_firstrun:
                features = mfcc_feat
                target = [0] * mfcc_feat.shape[0]
                for i in range(mfcc_feat.shape[0]):
                    target[row] = wav_dic[audio[0:-6]]
                    row += 1
                is_firstrun = False
            else:
                features = np.vstack((features, mfcc_feat))
                for i in range(mfcc_feat.shape[0]):
                    target = np.append(target, [wav_dic[audio[0:-6]]])
                    row += 1
        userList.write("Anonymous\n")
        userList.close()

        # Load user names
        userList = open(self.filepath+"/metadata.txt", "r")
        self.users = userList.read().split("\n")
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
        X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.33, random_state=42)


        self.mlp.fit(X_train,y_train)
        self.message += "\nTotal iteration run: %d" %self.mlp.n_iter_

        predictions = self.mlp.predict(X_test)

        from sklearn.metrics import classification_report,confusion_matrix
        self.message += "\n"
        self.message += classification_report(y_test, predictions, target_names=self.users)
        self.message += "\nconfusion_matrix\n"
        self.message += np.array_str(confusion_matrix(y_test, predictions))


        # Training is now complete, save the model
        pickle.dump(self.mlp, open(self.filepath+'/model.pkl','wb'))

        self.message += "\nModel is saved to %s/model.pkl" %self.filepath

        if self.verbose:
            print(self.message)

        # Load files
        self.NN = pickle.load(open(self.filepath+'/model.pkl','rb'))

        # Load user names
        userList = open(self.filepath+"/metadata.txt", "r")
        self.users = userList.read().split('\n')
        userList.close()

        return self.message
    # predict the output from given test audio files
    def test_predict(self):
        self.message = "Prediction result:"
        try:
            # Load files
            self.NN = pickle.load(open(self.filepath+'/model.pkl','rb'))
            # Load user names
            userList = open(self.filepath+"/metadata.txt", "r")
            self.users = userList.read().split('\n')
            userList.close()

            if self.verbose:
                print("Classes", self.NN.classes_)
                print("Output", self.NN.n_outputs_)
                print("Users are ", self.users)
        except FileNotFoundError:
            self.message += "\nmodel.pkl or metadata.txt file not found, make sure you have trained beforehand"
            return self.message

        try:
            wav_path = self.filepath+"/test/wav"
            wav_files = [f for f in os.listdir(wav_path) if f.endswith('.wav')]
            if(len(wav_files) < 1):
                return
                assert("No test wav files found at %s, Cancel operation" %wav_path)

            self.message += "\nTest file path: %s/test" %wav_path

            for wav_file in wav_files:
                self.message += "\n\nGiven wav file: %s" %wav_file
                fs, signal = wavfile.read(wav_path+"/"+ wav_file)
                # Compute output from feature
                output = self.predict(signal, fs)
                self.get_label(output)

        except (FileNotFoundError, IOError):
                self.message += "\nError file not found"
                return self.message

        if self.verbose:
            print(self.message)

        return self.message
    def predict(self, signal, fs):
        # remove silence
        signal = remove_silence(fs, signal)
        # signal = StandardScaler().fit_transform(signal)
        # Extract feature
        mfcc_feat = mfcc(signal, fs)
        if self.is_delta:
            mfcc_feat = delta(mfcc_feat, 2)

        return self.NN.predict(mfcc_feat)
    # Real time prediction
    def prediction(self, noise_level=230000):
        self.message = "Realtime prediction result"

        record_to_file(filename="test.wav",RECORD_SECONDS=2)
        fs, signal = wavfile.read("test.wav")
        os.remove("test.wav")

        if(silence_zone(signal, noise_level, verbose=self.verbose)):
            self.message += "\nSilent"
        else:
            output = self.predict(signal, fs)
            self.get_label(output)
        return self.message
    # Get a label from the given output of prediction
    def get_label(self, output,min_accuracy=0.6):
        counts = np.bincount(output)
        accuracy = (np.amax(counts)/output.shape[0])
        if self.verbose:
            self.message += "\nCounts: %s" %np.array_str(counts)
            self.message += "\nAccuracy: %.2f%%" %(accuracy*100)
        if accuracy < min_accuracy:
            self.message += "\nWho the fuck are you"
        else:
            self.message += "\nThe user is %s" %self.users[np.argmax(counts)]

        return self.message
    # set for delta mode
    def set_delta(self, delta):
        self.is_delta = delta
    # To set verbose mode to display additional information to console
    def set_verbose(self, verbose):
        self.verbose = verbose


def record_wav(filename, time=7):
    # record_to_file(filename="test.wav",RECORD_SECONDS=0.5)
    # inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    # inp.setchannels(1)
    # inp.setrate(8000)
    # inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    # inp.setperiodsize(1024)
    # w = wave.open(filename, 'w')
    # w.setnchannels(1)
    # w.setsampwidth(2)
    # w.setframerate(8000)
    #
    # for i in range(time): #~0.5 seconds
    #     l, data = inp.read()
    #     w.writeframes(data)
    #
    # w.close()
    pass

if __name__ == '__main__':
    nn = NeuralNetwork(filepath="../files")
    print_label("Training")
    print(nn.train())
    print_label("Testing from file...")
    print(nn.test_predict())
