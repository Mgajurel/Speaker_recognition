# Loop infinitely
while True:
	# Testing of each module is done from here
	print("***********************************************************")
	print("You can perform the following options:")
	print("1. Exctract the feature of files/test/test.wav")
	print("2. Test recording voice and save to files/wav/$filename.wav")
	print("3. Train the model with wav files located in files/wav")
	print("4. Predict from realtime voice signal")
	# Inform user way to cancel the operation
	print("   Ctrl + C to cancel")
	print("-----------------------------------------------------------")
	# Record the user choice
	choice = input("Enter you choice: ")
	# Now perform task according to the choice
	if choice == "1":
		print("Extracting the mfcc of the test file...")
		# Ask user if they want full list or not
		isFullList = input("Do you want full array to be shown(y/n)?: ")
		fullList = False
		if(isFullList == 'y'):
			fullList = True
		# File whose mfcc is to be detemined
		from scipy.io import wavfile
		fs, signal = wavfile.read("files/test/test.wav")
		# Calculate mfcc
		from feature.MFCC import MFCCExtractor
		mfcc = MFCCExtractor(fs, SHOW_AS_FULL_LIST=fullList);
		# Show mfcc on console
		print("MFCC of files/test/test.wav:")
		print(mfcc.extract(signal, "filename"))

	elif choice == "2":
		filename = input("Enter the filename: ")
		#TODO record using mic and save to files/wav/$filename
		print("Test recording to be done later and save as", filename)

	elif choice == "3":
		# TODO get all wav files at files/wav/*.wav
		# compute mfcc of each wav files
		# train the model using neural network and save them for later use
		print("Training to be done")

	elif choice == "4":
		# TODO record a sample audio default of < 0.9seconds
		# calculate its mfcc
		# pass it to the neural network
		# display the output to the console
		print("Prediction to be done")

	else:
		print("Invalid choice, please enter 1-4 only")
	input("Press any key to continue or Ctrl+C to cancel...")
