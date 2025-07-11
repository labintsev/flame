import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import ui_1
from model import Frame_thread

class Widget(QtWidgets.QMainWindow, ui_1.Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.thread = None
        self.PATH_TO_VIDEO = None

        self.setWindowTitle("FireController")
        self.setWindowIcon(QtGui.QIcon("img/icon.png"))
        self.setFixedSize(1000, 750)
        self.x_right.setIcon(QtGui.QIcon("img/rigth.png"))
        self.x_left.setIcon(QtGui.QIcon("img/left.png"))
        self.y_up.setIcon(QtGui.QIcon("img/up.png"))
        self.y_down.setIcon(QtGui.QIcon("img/down.png"))
        self.size_plus.setIcon(QtGui.QIcon("img/plus.png"))
        self.size_min.setIcon(QtGui.QIcon("img/minus.png"))

        self.view.setEnabled(False)
        self.setting.setEnabled(False)
        self.mask_value.setValue(140)
        self.result_file.setText("out")
        self.label_4.setText("Чувствительность маски равна 140")

        self.input_file.setReadOnly(True)
        self.camera.currentIndexChanged.connect(self.open_video_stream)
        self.view.clicked.connect(self.view_video)
        self.analize.clicked.connect(self.analize_video)
        self.mask.clicked.connect(self.toggle_draw_mask)
        self.stop.clicked.connect(self.stopVideo)
        self.mask_value.valueChanged.connect(self.getMask)
        self.x_left.clicked.connect(self.x_decrement)
        self.x_right.clicked.connect(self.x_increment)
        self.y_up.clicked.connect(self.y_increment)
        self.y_down.clicked.connect(self.y_decrement)
        self.size_plus.clicked.connect(self.size_increment)
        self.size_min.clicked.connect(self.size_decrement)

    def open_video_stream(self):
        if not self.camera.currentIndex():
            file = QtWidgets.QFileDialog.getOpenFileName(
                caption="Выбор файла",
                directory="c:\\",
                filter="All (*);;Exes (*.mov);;Exes (*.avi)",
            )
            if file[0]:
                self.PATH_TO_VIDEO = file[0]
                self.input_file.setText(self.PATH_TO_VIDEO)
                self.view.setEnabled(True)
        else:
            self.PATH_TO_VIDEO = int(self.camera.currentText())
            cap = cv2.VideoCapture(self.PATH_TO_VIDEO, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                self.input_file.setText("Камера не активна")
                self.view.setEnabled(False)
            else:
                self.input_file.setText("Камера активна")
                self.view.setEnabled(True)

    def view_video(self):
        self.thread = Frame_thread(self.PATH_TO_VIDEO)
        self.thread.changePixmap.connect(self.setImage)
        self.thread.start()
        self.setting.setEnabled(True)
        self.view.setEnabled(False)

    def analize_video(self):
        
        self.thread.Q = self.Qv.value()
        self.thread.S = self.ssg.value()
        self.thread.K = self.Koef.value()
        self.thread.Qp = (
            (self.H1.value() / 100) * self.Nmax_1.value()
            + (self.H2.value() / 100) * self.Nmax_2.value()
            + (self.H3.value() / 100) * self.Nmax_3.value()
        )
        self.thread.do_analize = not self.thread.do_analize
        self.thread.file_name = (
            self.result_file.text() + "." + self.expansion.currentText()
        )

    def toggle_draw_mask(self):
        self.thread.draw_mask = not self.thread.draw_mask

    def stopVideo(self):
        self.thread.running = False
        self.result_file.setEnabled(True)
        self.view.setEnabled(True)

    def setImage(self, image):
        self.video.setPixmap(QPixmap.fromImage(image))

    def getMask(self):
        self.thread.mask_treshold = self.mask_value.value()
        text = "Чувствительность маски равна " + str(self.mask_value.value())
        self.label_4.setText(text)

    def size_increment(self):
        if self.thread.x_end - self.thread.x_start > 150:
            self.thread.x_start += 75
            self.thread.x_end -= 75
        if self.thread.y_end - self.thread.y_start > 150:
            self.thread.y_start += 75
            self.thread.y_end -= 75

    def size_decrement(self):
        if self.thread.x_start > 0:
            self.thread.x_start -= 75
            self.thread.x_end += 75
            self.thread.delta_x = 0
        if self.thread.y_start > 0:
            self.thread.y_start -= 75
            self.thread.y_end += 75
            self.thread.delta_y = 0

    def x_increment(self):
        if self.thread.x_end < self.thread.width_0:
            self.thread.delta_x += 25
        if self.thread.x_end + self.thread.delta_x > self.thread.width_0:
            self.thread.delta_x -= 25

    def x_decrement(self):
        if self.thread.x_start > 0:
            self.thread.delta_x -= 25
        if self.thread.x_start + self.thread.delta_x < 0:
            self.thread.delta_x += 25

    def y_increment(self):
        if self.thread.y_start > 0:
            self.thread.delta_y -= 25
        if self.thread.y_start + self.thread.delta_y < 0:
            self.thread.delta_y += 25

    def y_decrement(self):
        if self.thread.y_end < self.thread.height_0:
            self.thread.delta_y += 25
        if self.thread.y_end + self.thread.delta_y > self.thread.height_0:
            self.thread.delta_y -= 25
