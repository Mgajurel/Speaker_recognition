from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QThread

if __name__ == '__main__':
    import sys
    sys.path.append("..")

from nn.neural_network import NeuralNetwork

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
        self.checkBox_verbose.clicked.connect(self.verbose_changed)
        self.checkBox_delta_mode.clicked.connect(self.delta_changed)

        # Show the form
        self.show()
        self.nn = NeuralNetwork(is_delta_mode=False, verbose=verbose)

    def verbose_changed(self):
        self.nn.set_verbose(self.checkBox_verbose.isChecked())

    def delta_changed(self):
        self.nn.set_delta(self.checkBox_delta_mode.isChecked())

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
        self.lbl_output.setText(self.output)
        if self.isPredicting:
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
    app = QtWidgets.QApplication(sys.argv)
    import os
    os.chdir("..")
    window = Ui(uipath="gui/user_interface.ui")
    sys.exit(app.exec_())
