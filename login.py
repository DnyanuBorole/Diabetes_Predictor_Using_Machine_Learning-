# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\QT\DIABETES\login.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from registration import *
from home import *





import sys


class lpage(object):

    def alertmsg(self, title, Message):
        self.warn = QtWidgets.QMessageBox()
        self.warn.setIcon(QtWidgets.QMessageBox.Information)
        self.warn.setWindowTitle(title)
        self.warn.setText(Message)
        self.warn.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.warn.setWindowIcon(QtGui.QIcon('error.png'))
        self.warn.exec_()

    def login(self):

        uid = self.lineEdit.text()
        # if uid == '' or uid == "null":
        # if uid == '' or uid == "null":
        #     self.alertmsg("ERROR", "Please Enter Your Username")

        passwd = self.lineEdit_2.text()
        # if passwd == '' or passwd == "null":
        #     self.alertmsg("ERROR", "Please Type Your Password")

        mydb = mysql.connector.connect(host="localhost", user="root", password="root", database='diabetes')
        cursor = mydb.cursor()
        query = "SELECT userid,password FROM `registration` WHERE userid = (%s) AND password = (%s) "
        cursor.execute(query, (uid, passwd,))

        if (len(cursor.fetchall()) > 0):
            # self.alertmsg("LoggedIn", "Login Successfully")
            self.Dialog = QtWidgets.QDialog()
            self.ui = home()
            self.ui.setupUi(self.Dialog)
            self.Dialog.show()

            # app.exec_()

        else:
            self.alertmsg("ERROR", "Please Enter Correct Username And Password")


    def reg(self):
        self.Dialog = QtWidgets.QDialog()
        self.ui = rpage(self.Dialog)
        self.ui.setupUi(self.Dialog)
        self.Dialog.show()

        # app.exec_()
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1095, 833)
        Dialog.setStyleSheet("QDialog{background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));}\n"
"")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(0, 0, 1095, 80))
        self.frame.setStyleSheet("background-color: qconicalgradient(cx:0, cy:0, angle:135, stop:0 rgba(255, 255, 0, 69), stop:0.375 rgba(255, 255, 0, 69), stop:0.423533 rgba(251, 255, 0, 145), stop:0.45 rgba(247, 255, 0, 208), stop:0.477581 rgba(255, 244, 71, 130), stop:0.518717 rgba(255, 218, 71, 130), stop:0.55 rgba(255, 255, 0, 255), stop:0.57754 rgba(255, 203, 0, 130), stop:0.625 rgba(255, 255, 0, 69), stop:1 rgba(255, 255, 0, 69));")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(340, 10, 441, 51))
        self.label.setStyleSheet("color: rgb(0, 0, 0);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.frame_2 = QtWidgets.QFrame(Dialog)
        self.frame_2.setGeometry(QtCore.QRect(310, 110, 451, 611))
        self.frame_2.setStyleSheet("background-color: rgb(183, 200, 199);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setMidLineWidth(4)
        self.frame_2.setObjectName("frame_2")
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setGeometry(QtCore.QRect(110, 50, 221, 41))
        self.label_4.setStyleSheet("background-color: rgb(135, 185, 190);")
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setLineWidth(5)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.lineEdit = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit.setGeometry(QtCore.QRect(235, 234, 152, 31))
        self.lineEdit.setStyleSheet("background-color: rgb(199, 200, 176);\n"
"font: 10pt \"MS Shell Dlg 2\";")
        self.lineEdit.setObjectName("lineEdit")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(90, 300, 101, 41))
        self.label_3.setObjectName("label_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(235, 304, 155, 31))
        self.lineEdit_2.setStyleSheet("background-color: rgb(199, 200, 176);\n"
"font: 10pt \"MS Shell Dlg 2\";")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setGeometry(QtCore.QRect(40, 520, 161, 41))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 rgba(0, 0, 0, 255), stop:0.05 rgba(14, 8, 73, 255), stop:0.36 rgba(28, 17, 145, 255), stop:0.6 rgba(126, 14, 81, 255), stop:0.75 rgba(234, 11, 11, 255), stop:0.79 rgba(244, 70, 5, 255), stop:0.86 rgba(255, 136, 0, 255), stop:0.935 rgba(239, 236, 55, 255));\n"
"color: rgb(240, 240, 240);\n"
"font: 12pt \"Algerian\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_2.setGeometry(QtCore.QRect(260, 520, 161, 41))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 rgba(0, 0, 0, 255), stop:0.05 rgba(14, 8, 73, 255), stop:0.36 rgba(28, 17, 145, 255), stop:0.6 rgba(126, 14, 81, 255), stop:0.75 rgba(234, 11, 11, 255), stop:0.79 rgba(244, 70, 5, 255), stop:0.86 rgba(255, 136, 0, 255), stop:0.935 rgba(239, 236, 55, 255));\n"
"color: rgb(240, 240, 240);\n"
"font: 12pt \"Algerian\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(100, 230, 101, 41))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton_2.clicked.connect(self.reg)
        # self.pushButton.clicked.connect(self.diabet)
        self.pushButton.clicked.connect(self.login)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:22pt; font-weight:600; color:#c82519;\">DIABETES PREDICTOR</span></p></body></html>"))
        self.label_4.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600; color:#000000;\">LOGIN</span></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">Password</span></p></body></html>"))
        self.pushButton.setText(_translate("Dialog", "LOGIN"))
        self.pushButton_2.setText(_translate("Dialog", "SIGNUP"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">User ID</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = lpage()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
