import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRect
import numpy as np
import cv2
from numba import njit, prange, jit
import time

opencv_palletes = {'autumn': 0, 'bone': 1, 'jet': 2, 'winter': 3, 'hot': 11, 'hsv': 9, 'pink': 10, 'ocean': 5, 'rainbow': 4, 'spring': 7, 'summer': 6, 'cool': 8, 'cividis': 17, 'twilight': 18, 'twilight_shifted': 19, 'turbo': 20}



class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit(event.x(), event.y())


@njit(parallel=True)
def generate_wave(array, x, y, step, beta, omega, wave_lambda):
    # value = Amplitude * e ^ (-Bt) * cos(wt + fi)
    height, width = array.shape
    ebt = np.exp(-beta*step)
    os = omega * step
    sw = step * wave_lambda
    for i in prange(width):
        for j in prange(height):
            distance = np.sqrt((x - i)**2 + (y - j)**2)
            if sw > distance:
                array[j][i] = 128 + (127 * ebt * np.cos(os + distance))


class ImageDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.x = 0
        self.y = 0
        self.t0 = -1
        self.beta = 0.3
        self.omega = 5
        self.wave_lambda = 100
        self.palette = 0
        self.initUI()

    def readValues(self):
        try:
            self.beta = float(self.inputFieldBeta.text())
            self.omega = float(self.inputFieldOmega.text())
            self.wave_lambda = float(self.inputFieldLambda.text())
        except ValueError:
            print('Excepted float values for beta and omega')
        self.palette = opencv_palletes[self.palette_selector.currentText()]

    def on_label_clicked(self, x, y):
        self.x = x
        self.y = y
        self.t0 = time.perf_counter()

    def initUI(self):
        self.inputLabelBeta = QLabel("β")
        self.inputFieldBeta = QLineEdit()
        self.inputFieldBeta.setText(str(self.beta))
        betaLayout = QHBoxLayout()
        betaLayout.addWidget(self.inputLabelBeta)
        betaLayout.addWidget(self.inputFieldBeta)

        self.inputLabelOmega = QLabel("ω")
        self.inputFieldOmega = QLineEdit()
        self.inputFieldOmega.setText(str(self.omega))
        omegaLayout = QHBoxLayout()
        omegaLayout.addWidget(self.inputLabelOmega)
        omegaLayout.addWidget(self.inputFieldOmega)

        self.inputLabelLambda = QLabel("λ")
        self.inputFieldLambda = QLineEdit()
        self.inputFieldLambda.setText(str(self.wave_lambda))
        lambdaLayout = QHBoxLayout()
        lambdaLayout.addWidget(self.inputLabelLambda)
        lambdaLayout.addWidget(self.inputFieldLambda)

        self.palette_selector = QComboBox()
        for palette in opencv_palletes.keys():
            self.palette_selector.addItem(palette)

        sliderLayout = QVBoxLayout()
        sliderLayout.setGeometry(QRect(0, 0, 200, 400))
        sliderLayout.addStretch(1)
        sliderLayout.addWidget(self.palette_selector)

        sliderLayout.addLayout(betaLayout)
        sliderLayout.addLayout(omegaLayout)
        sliderLayout.addLayout(lambdaLayout)

        self.button = QPushButton("Set")
        sliderLayout.addWidget(self.button)

        self.button.clicked.connect(self.readValues)
        sliderLayout.addStretch(1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateImage)
        self.timer.start(10)

        self.imageLabel = ClickableLabel(self)
        self.black_image = np.zeros((800,600), dtype=np.uint8)
        rgb_image = cv2.applyColorMap(self.black_image, cv2.COLORMAP_HSV)

        self.imageLabel.setPixmap(QPixmap.fromImage(QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)))
        self.imageLabel.clicked.connect(self.on_label_clicked)

        mainLayout = QHBoxLayout()

        mainLayout.addWidget(self.imageLabel)
        mainLayout.addLayout(sliderLayout)


        self.setLayout(mainLayout)

        self.setWindowTitle('Wave generator')
        self.setWindowIcon(QIcon('waveicon.png'))
        self.setGeometry(300, 300, 800, 600)

        self.show()

    def updateImage(self):

        if self.t0 == -1:
            return

        t_delta = time.perf_counter() - self.t0

        generate_wave(self.black_image, self.x, self.y, t_delta, self.beta, self.omega, self.wave_lambda)

        rgb_image = cv2.applyColorMap(self.black_image, self.palette)

        qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)

        self.imageLabel.setPixmap(QPixmap.fromImage(qimage))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageDisplay()
    sys.exit(app.exec_())