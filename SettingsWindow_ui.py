# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SettingsWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog.resize(558, 255)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(200, 200, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(100, 20, 30, 16))
        self.label.setObjectName("label")
        self.FolderLabel = QtWidgets.QLineEdit(Dialog)
        self.FolderLabel.setGeometry(QtCore.QRect(160, 20, 241, 20))
        self.FolderLabel.setToolTipDuration(2)
        self.FolderLabel.setObjectName("FolderLabel")
        self.BrowseButton = QtWidgets.QPushButton(Dialog)
        self.BrowseButton.setGeometry(QtCore.QRect(410, 20, 75, 23))
        self.BrowseButton.setObjectName("BrowseButton")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(60, 60, 81, 16))
        self.label_2.setObjectName("label_2")
        self.NamePrefixLabel = QtWidgets.QLineEdit(Dialog)
        self.NamePrefixLabel.setGeometry(QtCore.QRect(160, 60, 241, 20))
        self.NamePrefixLabel.setToolTipDuration(10)
        self.NamePrefixLabel.setObjectName("NamePrefixLabel")
        self.SampleRate = QtWidgets.QComboBox(Dialog)
        self.SampleRate.setGeometry(QtCore.QRect(160, 100, 41, 22))
        self.SampleRate.setObjectName("SampleRate")
        self.SampleRate.addItem("")
        self.SampleRate.addItem("")
        self.SampleRate.addItem("")
        self.SampleRate.addItem("")
        self.SampleRate.addItem("")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(43, 100, 91, 21))
        self.label_3.setObjectName("label_3")
        self.Range = QtWidgets.QComboBox(Dialog)
        self.Range.setGeometry(QtCore.QRect(160, 140, 91, 22))
        self.Range.setObjectName("Range")
        self.Range.addItem("")
        self.Range.addItem("")
        self.Range.addItem("")
        self.Range.addItem("")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(100, 140, 31, 21))
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Folder"))
        self.BrowseButton.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "Filename Prefix"))
        self.NamePrefixLabel.setToolTip(_translate("Dialog", "<>:\"/|?* not allowed"))
        self.NamePrefixLabel.setStatusTip(_translate("Dialog", "<>:\"/|?* not allowed"))
        self.SampleRate.setItemText(0, _translate("Dialog", "500"))
        self.SampleRate.setItemText(1, _translate("Dialog", "250"))
        self.SampleRate.setItemText(2, _translate("Dialog", "200"))
        self.SampleRate.setItemText(3, _translate("Dialog", "100"))
        self.SampleRate.setItemText(4, _translate("Dialog", "50"))
        self.label_3.setText(_translate("Dialog", "Sample Rate (kHz)"))
        self.Range.setItemText(0, _translate("Dialog", "BIP10VOLTS"))
        self.Range.setItemText(1, _translate("Dialog", "BIP5VOLTS"))
        self.Range.setItemText(2, _translate("Dialog", "BIP2VOLTS"))
        self.Range.setItemText(3, _translate("Dialog", "BIP1VOLTS"))
        self.label_4.setText(_translate("Dialog", "Range"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

