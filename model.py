import time
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage


class FlameModel:
    def __init__(self):
        pass

    def process_frame(self):
        pass

    def serialize(file_name, betta, UH, Qprib):
        """
        Saves measurement data and calculates error statistics for an experiment.

        Parameters:
            name_file (str): The base filename for saving results, extension is '.txt', saves as CSV
            betta (list or array-like): List of angle values (phi) for each measurement.
            UH (list or array-like): List of measured voltage values (Uн) corresponding to each angle.
            Qprib (float): Instrument error to be included in the total error calculation.

        Functionality:
            - Saves the measurement data (phi, Uн) to a file (CSV or Excel).
            - Calculates the arithmetic mean (X) of the measured values.
            - Computes the absolute error (Qабс), relative error (Qотн), and total error (Qполн).
            - Determines the confidence interval (Аверх, Анижн).
            - Writes a summary report with all calculated statistics to a separate text file.

        Output:
            - Data file with measurement results.
            - Text report file with error analysis and statistics.
        """
        result = {"phi": betta, "Uн": UH}
        if file_name[-3:] == "txt":
            file = open(file_name, "w")
            file.write("phi,Uн\n")
            for i in range(len(betta)):
                file.write(f"{betta[i]},{UH[i]}\n")
            file.close()
        elif file_name[-4:] == "xlsx":
            df = pd.DataFrame(result)
            df.to_excel(file_name)
        else:
            print('Unknown file format')

        X = sum(UH) / len(UH)
        Qapi = UH
        for i in range(len(Qapi)):
            Qapi[i] = abs(Qapi[i] - X)
        Qapi = sum(Qapi) / len(UH)
        Qotn = Qapi / X * 100
        Q = Qapi + Qprib
        Atop = X + Q
        Abot = X - Q
        with open(file_name[:-4] + "_отчет.txt", "w") as file:
            file.write(
                f"Всего снято показаний: {len(UH)}\n"
                f"Среднее арифметическое значение Х = {X}\n"
                f"Среднее абслолютное погрешности измерений Qабс = {Qapi}\n"
                f"Среднее относительное погрешности измерений Qотн = {Qotn} %\n"
                f"Оценка полной погрешности эксперимента Qполн = {Q}\n"
                f"Доверительный интервал: Аверх = {Atop}    Анижн = {Abot}\n"
            )



class Frame_thread(QThread):
    """
    QThread subclass for processing video frames in real-time.

    This class handles video capture from a file or camera, applies optional masking,
    analyzes frames to extract angles and voltages, and emits processed frames to the GUI.
    It also collects measurement data and triggers saving of results and error analysis.

    Attributes:
        changePixmap (pyqtSignal): Class attribute. Signal emitted with the processed QImage for display.
        nameFile (str): Output filename for saving results.
        source (str or int): Video file path or camera index.
        running (bool): Controls the main processing loop.
        mask (bool): Enables or disables mask processing.
        ValueMask (int): Threshold value for mask.
        analize (bool): Enables or disables analysis mode.
        Q (float): Experimental parameter for voltage calculation.
        S (float): Experimental parameter for voltage calculation.
        K (float): Experimental parameter for voltage calculation.
        Qp (float): Instrument error for error analysis.
        width_0 (int): Frame width.
        height_0 (int): Frame height.
        delta_x (int): Horizontal offset for region of interest.
        delta_y (int): Vertical offset for region of interest.
        x_start, x_end, y_start, y_end (int): Coordinates for region of interest.
    """
    changePixmap = pyqtSignal(QImage) 

    def __init__(self, source):
        """
        Initialize the Frame_thread.

        Args:
            source (str or int): Path to video file or camera index. 
        """
        super().__init__()
        
        self.file_name = None
        self.source = source

        self.running = True
        self.do_analize = False
        self.draw_mask = False
        self.mask_treshold = 140
        self.Q = 1e-5
        self.S = 1e-5
        self.K = 1
        self.Qp = 0

        self.width_0 = 0
        self.height_0 = 0

        self.delta_x = 0
        self.delta_y = 0

        self.x_start, self.x_end = 0, 0
        self.y_start, self.y_end = 0, 0

        # Todo adjust delay in UI
        self.delay = 0.3

    def run(self):
        """
        Main thread loop for capturing and processing video frames.

        - Captures frames from the video source.
        - Applies region of interest and optional masking.
        - If analysis is enabled, extracts angle and voltage data.
        - Emits processed frames to the GUI.
        - Saves measurement data and error analysis after processing.
        """
        cap = cv2.VideoCapture(self.source)
        self.running = True
        ret, frame = cap.read()
        betta = []
        UH = []
        self.width_0 = frame.shape[1]
        self.height_0 = frame.shape[0]
        self.x_end = self.width_0
        self.y_end = self.height_0


        while self.running:
            ret, frame = cap.read()
            time.sleep(self.delay)
            if not ret:
                cap.release()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ch = frame.shape[2]
            bytes_per_line = ch * self.width_0
            frame = frame[
                self.y_start + self.delta_y : self.y_end + self.delta_y,
                self.x_start + self.delta_x : self.x_end + self.delta_x,
            ]
            frame = cv2.resize(frame, (self.width_0, self.height_0))

            if self.draw_mask:
                frame = frame[:, :, 2]
                frame = (frame > self.mask_treshold).astype(np.uint8) * 255
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            if self.do_analize:
                p1, p2, phi = self.extract_angle(frame)
                cv2.line(frame, p1, p2, (255, 0, 0), 5)
                if 90 > phi > 60:
                    betta.append(phi)
                    UH.append(
                        np.cos(phi * np.pi / 180)
                        * ((self.Q * 16.667) / (self.S + 1e-5))
                        * self.K
                    )
            image = QImage(frame.data, self.width_0, self.height_0, bytes_per_line, 13)
            image = image.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(image)

        if len(betta):
            FlameModel.serialize(self.file_name, betta, UH, self.Qp)

    def extract_angle(self, frame, channel=2):
        """
        Analyze a single frame to extract feature points and calculate the angle.

        Args:
            frame (np.ndarray): The input video frame (RGB).
            channel (int): Color channel to use for mask (default: 2, red).

        Returns:
            tuple: (p1, p2, phi)
                p1 (tuple): First feature point (x, y).
                p2 (tuple): Second feature point (x, y).
                phi (float): Calculated angle in degrees.
        """
        img_plane = frame[:, :, channel]
        img_mask = (img_plane > self.mask_treshold).astype(np.uint8)
        x = np.argmax(np.sum(img_mask, axis=0))
        y = np.argmax(np.sum(img_mask, axis=1))
        h, w = img_mask.shape
        p1 = x, np.argmax(img_mask[:, x])
        p2 = w - np.argmax(img_mask[y, ::-1]), y
        # print(f'p1={p1}, p2={p2}')
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        phi = 90 - np.arctan(dx / dy) * (180 / np.pi)
        return p1, p2, phi

    def stop(self):
        """
        Stop the frame processing thread.
        """
        self.running = False
