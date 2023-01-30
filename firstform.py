
from PyQt5 import QtCore, QtGui, QtWidgets 
from window import Ui_testWindow
from window2 import Ui_window2

class Ui_MainWindow(object):
    
    def trainopen(self):
        self.window= QtWidgets.QMainWindow()
        self.ui = Ui_window2()
        self.ui.setupUi(self.window)
        MainWindow.hide()
        self.window.show()
    
    def generateopen(self):
        self.window= QtWidgets.QMainWindow()
        self.ui = Ui_testWindow()
        self.ui.setupUi(self.window)
        MainWindow.hide()
        self.window.show()
    
        
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(270, 330, 251, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.trainopen)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(270, 420, 251, 51))
        self.pushButton_2.clicked.connect(self.generateopen)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 30, 591, 171))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 220, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
  
    
        
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Train"))
        self.pushButton_2.setText(_translate("MainWindow", "Generate"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:28pt;\">Welcome to our Project!</span></p><p align=\"center\"><span style=\" font-size:28pt;\"> Parent Face Detector</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Would Like to Train or Generate?"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
