def print_info(info):
	dash = int((80-len(info))/2)
	print("-"*dash, info, "-"*dash)

def print_label(text, character="*"):
	star = int((80-len(text))/2)
	print(character*star, text, character*star)

def print_footer():
	print("-"*82)

# Loop infinitely
while True:
	# Testing of each module is done from here
	print_label("Main Menu")
	print("You can perform the following options:")
	print("1. Exctract the feature of files/test/test.wav")
	print("2. Test recording voice and save to files/wav/$filename.wav")
	print("3. Train the model with wav files located in files/wav")
	print("4. Predict from realtime voice signal")
	print("e. To exit")
	# Inform user way to cancel the operation
	print("   Ctrl + C to cancel")
	print_footer()
	# Record the user choice
	choice = input("Enter you choice->")
	if choice == "1":
		print_label("Entering MFCC feature extraction mode")
		print_info("Converting wav to CSV")
		from feature.utils import wavsToCsv
		wavsToCsv("files/test/wav", "files/test/csv")
		print_info("Test feature files saved at files/test/csv")

	elif choice == "2":
		print_label("Entering Test record mode")
		filename = input("Enter the filename: ")
		#TODO record using mic and save to files/wav/$filename
		print("Test recording to be done later and save as", filename)

	elif choice == "3":
		print_label("Entering Training mode")
		toCsv = input("Do you want to convert all wav files to csv files(y/n)?")
		if toCsv == 'y':
			print_info("Converting wav to CSV")
			from feature.utils import wavsToCsv
			wavsToCsv("files/wav", "files/csv", verbose=True)
		else:
			print_info("Convert to csv cancelled")
		print_footer();

		# Training the network using neural network, this may take a while
		toTrain = input("Do you want to train the network, this may take a while(y/n)")
		if toTrain == 'y':
			print_info("Training started.")
			from model_training.TrainTheModel import training
			training("files/csv")
		else:
			print_info("Training cancelled.")


	elif choice == "4":
		print_label("Entering Prediction mode")
		filename = input("Enter filename of wav file at files/test/wav to test: ")
		from scipy.io import wavfile		
		try:
			fs, signal = wavfile.read("files/wav/" + filename)
			from feature.MFCC import MFCCExtractor
			mfcc = MFCCExtractor(fs)

			from model_training.TrainTheModel import Prediction
			prediction = Prediction()

			print_label("Prediction result:")
			print("File:", filename)
			prediction.predict(mfcc.extract(signal))

		except (FileNotFoundError, IOError):
			print("Wrong file or file path")

		# from model_training.TrainTheModel import prediction,csv_extractor
		# filepath = input("Enter file path to test: ")
		# feat = csv_extractor(filepath)	#files/csv/test_suren_1_mfcc.csv
		# # print(feat)
		# prediction(feat)

	elif choice == 'e':
		break

	else:
		print("Invalid choice, please enter 1-4 only")

	print_footer()
	if input("Press any key to continue or e to exit: ") == 'e':
		break