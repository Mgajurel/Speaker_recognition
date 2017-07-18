from PyQt5 import uic, QtWidgets, QtGui, QtCore
import sys
import shutil

import pyaudio
from recorder import Recorder
from utils import read_wav, write_wav, time_str, monophonic


d=True

def dprint(msg):
    if(d):
        print(msg)

FORMAT=pyaudio.paInt16
NPDtype = 'int16'
NAMELIST = ['Nobody']

class Main(QtWidgets.QDialog):
    CONV_INTERVAL = 0.4
    CONV_DURATION = 1.5
    CONV_FILTER_DURATION = CONV_DURATION
    FS = 8000
    TEST_DURATION = 3

    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi('user_interface.ui', self)
        #Recorder
        self.recorder = Recorder(channels=1, rate=8000); #Mono channel, 8000Mhz sample rate

        #UI Initializes
        self.userdata =[]
        self.loadUsers()
        self.Userchooser.currentIndexChanged.connect(self.showUserInfo)
        self.ClearInfo.clicked.connect(self.clearUserInfo)
        self.UpdateInfo.clicked.connect(self.updateUserInfo)
        self.UploadImage.clicked.connect(self.upload_avatar)

        self.enrollRecord.clicked.connect(self.start_enroll_record)
        self.stopEnrollRecord.clicked.connect(self.stop_enroll_record)
        self.enrollFile.clicked.connect(self.enroll_file)
        self.enroll.clicked.connect(self.do_enroll)
        self.startTrain.clicked.connect(self.start_train)
        self.dumpBtn.clicked.connect(self.dump)
        self.loadBtn.clicked.connect(self.load)

        #Default user image
        self.setProfilePic("avatar/test.jpg")
        self.show()

    def load(self):
        dprint("Load")

    def dump(self):
        dprint("dump")

    def start_train(self):
        dprint("Start Train")

    def do_enroll(self):
        dprint("Do enroll")

    def enroll_file(self):
        dprint("Open File")
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open Wav File", "", "File (*.wav)")
        image_path = str(fname[0])
        print(image_path)
        if not image_path:
            return
        self.enrollFileName.setText(image_path)
        fs, signal = read_wav(image_path)
        signal = monophonic(signal)
        self.enrollWav = (fs, signal)

    def stop_record(self):
        self.stopped = True
        self.reco_th.wait()
        self.timer.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        self.status("Record stopeed")

    def stop_enroll_record(self):
        dprint("Stop enroll record")
        self.stop_record()
        print (self.recordData[:300])
        signal = np.array(self.recordData, dtype=NPDtype)
        self.enrollWav = (Main.FS, signal)

        # TODO To Delete
        write_wav('enroll.wav', *self.enrollWav)

    def start_enroll_record(self):
        self.enrollWav = None
        self.enrollFileName.setText("")
        self.start_record()
        dprint("Start enroll record")


    def loadUsers(self):
        dprint("Load Users")
        with open("avatar/metainfo.txt") as db:
            for line in db:
                tmp = line.split()
                print(tmp[0])
                self.userdata.append(tmp)
                self.Userchooser.addItem(tmp[0])


    def clearUserInfo(self):
        dprint("Clear User Info")
        self.Username.setText("")
        self.Userage.setValue(22)
        self.Usersex.setCurrentIndex(0)
        self.setProfilePic(self.avatarname)

    def updateUserInfo(self):
        dprint("Update User Info")
        userindex = self.Userchooser.currentIndex() - 1
        u = self.userdata[userindex]
        u[0] = (self.Username.displayText())
        #save the image as this Username
        shutil.copy2("./avatar/temp.jpg", "./avatar/"+u[0]+".jpg")
        u[1] = self.Userage.value()
        if self.Usersex.currentIndex():
            u[2] = 'F'
        else:
            u[2] = 'M'
        with open("avatar/metainfo.txt","w") as db:
            for user in self.userdata:
                for i in range(3):
                    db.write(str(user[i]) + " ")
                db.write("\n")

    def writeuserdata(self):
        with open("avatar/metainfo.txt","w") as db:
            for user in self.userdata:
                for i in range (0,4):
                    db.write(str(user[i]) + " ")
                db.write("\n")

    def addUserInfo(self):
        for user in self.userdata:
            if user[0] == unicode(self.Username.displayText()):
                return
        newuser = []
        newuser.append(unicode(self.Username.displayText()))
        newuser.append(self.Userage.value())
        if self.Usersex.currentIndex():
            newuser.append('F')
        else:
            newuser.append('M')
        if self.avatarname:
            shutil.copy(self.avatarname, 'avatar/' + user[0] + '.jpg')
        self.userdata.append(newuser)
        self.writeuserdata()
        self.Userchooser.addItem(unicode(self.Username.displayText()))

    def showUserInfo(self):
        for user in self.userdata:
            if self.userdata.index(user) == self.Userchooser.currentIndex() - 1:
                self.Username.setText(user[0])
                self.Userage.setValue(int(user[1]))
                if user[2] == 'F':
                    self.Usersex.setCurrentIndex(1)
                else:
                    self.Usersex.setCurrentIndex(0)
                self.setProfilePic(self.get_avatar([user[0]]))
                self.Userimage.setPixmap(self.get_avatar(user[0]))


    def upload_avatar(self):
        dprint("Upload Avatar")
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open jpg File", "", "File (*.jpg)")
        image_path = str(fname[0])
        print(image_path)
        if not image_path:
            return
        shutil.copy2(image_path, "./avatar/temp.jpg")
        self.setProfilePic(image_path)

    def get_avatar(self, username):
        p = self.avatars.get(str(username), None)
        if p:
            return p
        else:
            return self.defaultimage

    def setProfilePic(self, path):
        self.avatarname = path
        image_profile = QtGui.QImage(path) #QImage object
        image_profile = image_profile.scaled(100,100, transformMode=QtCore.Qt.SmoothTransformation) # To scale image for example and keep its Aspect Ration
        self.Userimage.setPixmap(QtGui.QPixmap.fromImage(image_profile))

    ############ RECORD
    def start_record(self):
        self.pyaudio = pyaudio.PyAudio()

        self.recordData = []
        self.stream = self.pyaudio.open(format=FORMAT, channels=1, rate=Main.FS,
                        input=True, frames_per_buffer=1)
        self.stopped = False
        self.reco_th = RecorderThread(self)
        self.reco_th.start()

        self.timer.start(1000)
        self.record_time = 0
        self.update_all_timer()

    def update_all_timer(self):
        s = time_str(self.record_time)
        self.enrollTime.setText(s)
        self.recoTime.setText(s)
        self.convTime.setText(s)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    sys.exit(app.exec_())
