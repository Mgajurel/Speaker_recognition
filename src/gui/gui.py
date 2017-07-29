from PyQt5 import uic, QtWidgets
from nn.neural_network import NeuralNetwork

import sys
import time

from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal)

def print_label(text, character="*"):
    star = int((80-len(text))/2)
    print(character*star, text, character*star)


class TrainThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.main.start_train_thread()

class PredictionThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.main.start_predict_thread()
        
class TestPredictionThread(QThread):
    def __init__(self, main):
        QThread.__init__(self)
        self.main = main

    def run(self):
        self.main.start_test_predict_thread()

class Ui(QtWidgets.QDialog):
    def __init__(self, uipath="user_interface.ui", verbose=False):
        super(Ui, self).__init__()
        uic.loadUi(uipath, self)
        self.status = ""
        self.finished = pyqtSignal()
        self.output = ""
        self.isPredicting=False

        #Thread
        self.train_th = TrainThread(self)
        self.train_th.finished.connect(self.train_fin)

        self.predict_th = PredictionThread(self)
        self.predict_th.finished.connect(self.predict_fin)

        self.test_predict_th = TestPredictionThread(self)
        self.test_predict_th.finished.connect(self.test_predict_fin)

        # UI Initializes
        self.btn_train.clicked.connect(self.start_train)
        self.btn_predict.clicked.connect(self.start_predict)
        self.btn_test_predict.clicked.connect(self.start_test_predict)

        # Show the form 
        self.show()
        self.nn = NeuralNetwork(is_delta_mode=False, verbose=verbose)

    def start_train(self):
        if(self.status == "training" or self.status != ""):
            print("Already", self.status)
            return
        self.status = "training"
        self.lbl_output.setText("Training started...")
        print("Training started")
        self.train_th.start()
        self.btn_train.setEnabled(False)

    def start_train_thread(self):
        self.output = self.nn.train()

    def train_fin(self):
        print("Training finished.")
        self.lbl_output.setText(self.output)
        self.status=""
        self.btn_train.setEnabled(True)

    def start_predict(self):
        if self.isPredicting:
            # Already predicting so stop prediction
            self.isPredicting = False
            self.predict_th.wait()  #Wait to complete the thread            
            self.btn_predict.setText("Predict")
            self.lbl_output.setText("Prediction stopped.")

        else:
            # Prediction is not started start prediction
            self.isPredicting = True
            self.btn_predict.setText("Stop Prediction")
            self.predict_th.start()

    def start_predict_thread(self):
        self.output = self.nn.prediction()

    def predict_fin(self):
        if self.output == None:
            self.lbl_output.setText("Anynomous")
        else:
            self.lbl_output.setText("The user is "+ self.output)
            self.output="Anonymous"

        if self.isPredicting:
            self.predict_th.wait()
            self.predict_th.start()
        else:
            self.lbl_output.setText("Prediction complete")

    def start_test_predict(self):
        self.btn_test_predict.setEnabled(False)
        self.lbl_output.setText("Test Predict")
        self.test_predict_th.start()

    def start_test_predict_thread(self):
        self.output = self.nn.test_predict()

    def test_predict_fin(self):
        self.lbl_output.setText(self.output)
        self.btn_test_predict.setEnabled(True)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())
