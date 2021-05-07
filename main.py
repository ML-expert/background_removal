import cv2
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtCore import *
import sys
from inference import run

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('main.ui', self)

        self.initUI()
        self.mode = 0
        self.setWindowTitle("Remove Background")
        self.show()


    def initUI(self):
        self.btnOpenImg = self.findChild(QtWidgets.QPushButton, 'btnOpenImage')
        self.btnOpenImg.clicked.connect(self.OnBtnOpenImg)

        self.btnStart = self.findChild(QtWidgets.QPushButton, 'btnStart')
        self.btnStart.clicked.connect(self.OnBtnStart)

        self.editWidth = self.findChild(QtWidgets.QLineEdit, 'editWidth')
        self.editHeight = self.findChild(QtWidgets.QLineEdit, 'editHeight')
        self.imageBox = self.findChild(QtWidgets.QLabel, 'imageBox')


    def OnBtnStart(self):
        image = run(self.filePath)

        w_scale = self.imageBox.width() / image.shape[1]
        h_scale = self.imageBox.height() / image.shape[0]

        if w_scale > h_scale:
            imageToDisplay = cv2.resize(image,
                                        (int(image.shape[1] * h_scale), int(image.shape[0] * h_scale)))
        else:
            imageToDisplay = cv2.resize(image,
                                        (int(image.shape[1] * w_scale), int(image.shape[0] * w_scale)))

        imageToDisplay = QtGui.QImage(imageToDisplay.data, imageToDisplay.shape[1], imageToDisplay.shape[0],
                                      imageToDisplay.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()

        self.imageBox.setPixmap(QtGui.QPixmap.fromImage(imageToDisplay))


    def OnBtnOpenImg(self):
        self.filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', './', "Image files (*.jpg *.png *.gif)")
        self.fileName = QFileInfo(self.filePath).fileName()

        if self.filePath is "":
            return

        self.image = cv2.imread(self.filePath)

        w_scale = self.imageBox.width() / self.image.shape[1]
        h_scale = self.imageBox.height() / self.image.shape[0]

        if w_scale > h_scale:
            imageToDisplay = cv2.resize(self.image, (int(self.image.shape[1] * h_scale), int(self.image.shape[0] * h_scale)))
        else:
            imageToDisplay = cv2.resize(self.image, (int(self.image.shape[1] * w_scale), int(self.image.shape[0] * w_scale)))

        imageToDisplay = QtGui.QImage(imageToDisplay.data, imageToDisplay.shape[1], imageToDisplay.shape[0],
                                 imageToDisplay.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()

        self.imageBox.setPixmap(QtGui.QPixmap.fromImage(imageToDisplay))


    def exit(self):
        QtCore.QCoreApplication.instance().quit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()