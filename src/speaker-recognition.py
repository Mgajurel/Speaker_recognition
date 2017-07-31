def print_info(info):
	dash = int((80-len(info))/2)
	print("-"*dash, info, "-"*dash)

def print_label(text, character="*"):
	star = int((80-len(text))/2)
	print(character*star, text, character*star)

def print_footer():
	print("-"*82)

from nn.neural_network import NeuralNetwork
nn = NeuralNetwork(is_delta_mode=False, verbose = True)


# Loop infinitely
while True:
	# Testing of each module is done from here
	print_label("Main Menu")
	print("You can perform the following options:")
	print("1. Train the model with wav files located in files/wav")
	print("2. Test the output with given path to wav files")
	print("3. GUI interface")
	print("4. Realtime cli voice recognition")
	print("e. To exit")
	# Inform user way to cancel the operation
	print("   Ctrl + C to cancel")
	print_footer()
	# Record the user choice
	choice = input("Enter you choice->")
	if choice == "1":
		print_label("Entering Training mode")
		nn.train()

	elif choice == "2":
		print_label("Entering Prediction mode")
		nn.test_predict()

	elif choice == '3':
		print_label("Entering GUI mode")
		from PyQt5 import uic, QtWidgets
		from gui.gui import Ui
		import sys
		app = QtWidgets.QApplication(sys.argv)
		window = Ui(uipath="gui/user_interface.ui", verbose=False)
		sys.exit(app.exec_())

	elif choice == "4":
		print_label("Entering Prediction mode")
		while True:
			user = nn.prediction()
			print("Press CTRL+C to cancel")
			print("The user is %s" %user)
		# try:
		#
		# except:
		# 	print("Prediction stopped.")

	elif choice == 'e':
		break

	else:
		print("Invalid choice, please enter 1-4 only")

	print_footer()
	if input("Press any key to continue or e to exit: ") == 'e':
		break
