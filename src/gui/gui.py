from PyQt5 import uic, QtWidgets, QtGui, QtCore
import sys
import shutil

d=True

def dprint(msg):
    if(d):
        print(msg)

class Ui(QtWidgets.QDialog):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('user_interface.ui', self)

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
        dprint("Enroll file")

    def stop_enroll_record(self):
        dprint("Stop enroll record")

    def start_enroll_record(self):
        dprint("Start enroll record")


    def loadUsers(self):
        dprint("Load Users")
        with open("avatar/metainfo.txt") as db:
            for line in db:
                tmp = line.split()
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())
