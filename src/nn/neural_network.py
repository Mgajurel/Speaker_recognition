from scipy.io import wavfile
from feature.MFCC import mfcc
from feature.MFCC import delta
import os
import numpy as np
import pickle

debug = True

def print_label(text, character="*"):
	star = int((80-len(text))/2)
	print(character*star, text, character*star)

def dprint(message):
	if debug:
		print(message)

def confusion_matrix(y_test, predictions):
	correct = 0
	incorrect = 0
	for i in range(y_test.shape[0]):
		if(y_test[i] == predictions[i]).all():
			correct += 1
		else:
			incorrect += 1
	accuracy = correct / predictions.shape[0] * 100
	print("Accurate =", correct)
	print("Incorrect =", incorrect)
	print("Accuracy = %.2f%%" %round(accuracy , 2))

class NeuralNetwork:
	def __init__(self, filepath="files", is_delta_mode=False, verbose=True, accuracy=0.2):
		self.verbose = verbose
		self.filepath = filepath
		self.is_delta = is_delta_mode
		print("Delta Mode enable = ", is_delta_mode)

	def train(self):
		# File path
		wavpath=self.filepath+"/wav"
		dprint("Wav files from: " + wavpath)

		# List all the filenames of wav directory
		wav_files = [f for f in os.listdir(wavpath) if f.endswith('.wav')]
		n = len(wav_files)
		if(n < 1):
			print("No wav files found at " + wavpath + ". Cancelling operation")
			return

		features = None
		target = None
		is_firstrun = True
		row = 0
		col = 0
		userList = open("files/metadata.txt", "w")

		# Loop until all files are converted to csv
		for audio in wav_files:
			sample_rate, signal = wavfile.read(wavpath+"/"+audio)
			mfcc_feat = mfcc(signal, sample_rate)
			
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

		print_label("Data sets", character="-")
		features = features.astype(np.float)
		print("Features:", features.shape)
		print(features)
		print("\nTarget:", target.shape)
		print(target)

		# Features and target are made now train them

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

		mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30),
		    max_iter=10000, tol=1e-6)

		mlp.fit(X_train,y_train)
		print("Total iteration run:", mlp.n_iter_)

		predictions = mlp.predict(X_test)   

		# Display confusion matrix
		confusion_matrix(y_test, predictions)

		print("Neural network training finish")

		# Training is now complete, save the model
		pickle.dump(mlp, open(self.filepath+'/model.pkl','wb'))
		print("Model is saved to "+self.filepath+'/model.pkl')

	# predict the output from a given test audio files
	def test_predict(self):
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
				print("No test wav files found at", wav_path)
				print("Cancel operation")
				return

			print("Test file path: " + self.filepath+"/test")

			for wav_file in wav_files:
				fs, signal = wavfile.read(wav_path+"/"+ wav_file)				
				output = self.predict(signal, fs)
				print("Given wav file:", wav_file)		
				print("The user is", self.get_label(output))
				
		except (FileNotFoundError, IOError):
				print("Wrong file or file path")

	# Predict from a given wav signal
	def predict(self, signal, fs):
		mfcc_feat = mfcc(signal, fs)

		if self.is_delta:
			mfcc_feat = delta(mfcc_feat, 2)
			
		output = self.NN.predict(mfcc_feat)
		return output

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
			print("Prection Outcome")
			print("Count:", count_array)
			print("Accuracy = %.2f%%" %round(accuracy, 2))

		return label
