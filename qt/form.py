# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(577, 305)
        self.Addbutton = QtWidgets.QPushButton(Form)
        self.Addbutton.setGeometry(QtCore.QRect(180, 140, 51, 51))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.Addbutton.setFont(font)
        self.Addbutton.setIconSize(QtCore.QSize(40, 40))
        self.Addbutton.setObjectName("Addbutton")
        self.Equalbutton = QtWidgets.QPushButton(Form)
        self.Equalbutton.setGeometry(QtCore.QRect(360, 140, 51, 51))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.Equalbutton.setFont(font)
        self.Equalbutton.setIconSize(QtCore.QSize(40, 40))
        self.Equalbutton.setObjectName("Equalbutton")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(210, 50, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAutoFillBackground(False)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(80, 140, 61, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(260, 140, 61, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.result = QtWidgets.QLineEdit(Form)
        self.result.setGeometry(QtCore.QRect(440, 140, 61, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.result.setFont(font)
        self.result.setObjectName("result")

        self.retranslateUi(Form)
        self.Addbutton.clicked.connect(Form.add)
        self.Equalbutton.clicked.connect(Form.equal)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.Addbutton.setText(_translate("Form", "+"))
        self.Equalbutton.setText(_translate("Form", "="))
        self.label.setText(_translate("Form", "简易加法器"))

