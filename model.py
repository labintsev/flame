import os
import time
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QImage
import logging

# Setup logger
logger = logging.getLogger("flame_logger")
logger.setLevel(logging.INFO)
os.makedirs("output", exist_ok=True)
file_handler = logging.FileHandler(os.path.join("output", "log.txt"), encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Experiment(QObject):
    """
    Handles experimental parameters, data collection, angle extraction, and result serialization.

    This class stores experiment parameters, accumulates measurement data (angles and voltages),
    provides methods to update parameters, extract angles from frames, and serialize results
    with error analysis to files.
    """

    def __init__(self, out_file_name=None, source_name=None):
        """
        Initialize the Experiment object with default parameters and empty data lists.
        """
        super(Experiment, self).__init__()
        self.file_name = out_file_name
        self.source_name = source_name
        self.Q = 1e-5
        self.S = 1e-5
        self.K = 1
        self.Qp = 0
        self.Qprib = 0
        self.betta = []
        self.UH = []
        self.labels = []

        # Create an inference session
        self.session = ort.InferenceSession("convnet_0.onnx")
        logger.info("Experiment initialized with file_name=%s, source_name=%s", out_file_name, source_name)

    def update_params(self, Q, S, K, H1, H2, H3, Nmax_1, Nmax_2, Nmax_3):
        """
        Update experiment parameters and calculate instrument error.

        Args:
            Q (float): Experimental parameter Q.
            S (float): Experimental parameter S.
            K (float): Experimental parameter K.
            H1 (float): Error coefficient for Nmax_1.
            H2 (float): Error coefficient for Nmax_2.
            H3 (float): Error coefficient for Nmax_3.
            Nmax_1 (float): Maximum value for first measurement.
            Nmax_2 (float): Maximum value for second measurement.
            Nmax_3 (float): Maximum value for third measurement.
        """
        self.Q = Q
        self.S = S
        self.K = K
        self.Qp = (H1 / 100) * Nmax_1 + (H2 / 100) * Nmax_2 + (H3 / 100) * Nmax_3

    def extract_angle(self, frame, mask_treshold, channel=2):
        """
        Analyze a single frame to extract feature points and calculate the angle.

        Args:
            frame (np.ndarray): The input video frame (RGB).
            mask_treshold (int): Threshold value for mask.
            channel (int): Color channel to use for mask (default: 2, red).

        Returns:
            tuple: (p1, p2, phi)
                p1 (tuple): First feature point (x, y).
                p2 (tuple): Second feature point (x, y).
                phi (float): Calculated angle in degrees.

        Side Effects:
            Appends valid angle and voltage measurements to self.betta and self.UH.
        """
        img_plane = frame[:, :, channel]
        img_mask = (img_plane > mask_treshold).astype(np.uint8)
        x = np.argmax(np.sum(img_mask, axis=0))
        y = np.argmax(np.sum(img_mask, axis=1))
        h, w = img_mask.shape
        p1 = x, np.argmax(img_mask[:, x])
        p2 = w - np.argmax(img_mask[y, ::-1]), y
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        phi = 90 - np.arctan(dx / dy) * (180 / np.pi)
        if 90 > phi > 60:
            self.betta.append(phi)
            self.UH.append(
                np.cos(phi * np.pi / 180)
                * ((self.Q * 16.667) / (self.S + 1e-5))
                * self.K
            )
        return p1, p2, phi

    def predict_label(self, frame):
        # Preprocess the input frame
        frame = cv2.resize(frame, (1024, 1024))
        frame = frame.transpose(2, 0, 1)  # Change data layout to (C, H, W)
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        # Normalize to [-1, 1]
        frame = (frame.astype(np.float32) / 255.0 - 0.5) * 2.0
        # Run inference
        outputs = self.session.run(None, {"input": frame})
        # Get the predicted label
        label = np.argmax(outputs[0], axis=1)[0]
        self.labels.append(label)

    def serialize(self):
        """
        Saves measurement data and calculates error statistics for an experiment.

        Args:
            file_name (str): The base filename for saving results. If the extension is '.txt', saves as CSV;
                             if '.xlsx', saves as Excel; otherwise, prints an error.

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
        if len(self.betta) == 0:
            logger.warning("No measurements result")
            return

        if len(self.UH) == len(self.betta) == len(self.labels):
            result = {"phi": self.betta, "Uн": self.UH, "label": self.labels}
            df = pd.DataFrame(result)
            df.to_csv(os.path.join("output", self.file_name), index=False)
            logger.info("Saved results to %s", os.path.join("output", self.file_name))
        else:
            logger.error("Unequal lengths of measurements, cannot save results. Lengths: betta=%d, UH=%d, labels=%d",
                         len(self.betta), len(self.UH), len(self.labels))
            return
        # todo refactor computations to other method and numpy
        X_mean = np.mean(self.UH)
        Qapi = self.UH
        for i in range(len(Qapi)):
            Qapi[i] = abs(Qapi[i] - X_mean)
        Qapi = sum(Qapi) / len(self.UH)
        Qotn = Qapi / X_mean * 100
        Q = Qapi + self.Qprib
        Atop = X_mean + Q
        Abot = X_mean - Q

        reportname = self.file_name[:-4] + "_report.txt"

        with open(os.path.join("output", reportname), "w", encoding="utf-8") as file:
            file.write(
                f"""
Источник данных: {self.source_name}
Всего снято показаний: {len(self.UH)} 
Среднее арифметическое значение Х = {X_mean} 
Среднее абслолютное погрешности измерений Q_abc = {Qapi} 
Среднее относительное погрешности измерений Q_rel = {Qotn} 
Оценка полной погрешности эксперимента Q_total = {Q}
Доверительный интервал: Аверх = {Atop}    Анижн = {Abot}
"""
            )
        logger.info("Report saved to %s", os.path.join("output", reportname))

class FrameThread(QThread):
    """
    QThread subclass for real-time video frame processing and analysis.

    This class captures video frames from a file or camera, applies optional masking,
    analyzes frames to extract angles and voltages, and emits processed frames to the GUI.
    It accumulates measurement data and saves results and error analysis after processing.

    Attributes:
        changePixmap (pyqtSignal): Signal emitted with the processed QImage for display.
        file_name (str): Output filename for saving results.
        source (str or int): Video file path or camera index.
        running (bool): Controls the main processing loop.
        do_analize (bool): Enables or disables analysis mode.
        draw_mask (bool): Enables or disables mask drawing.
        mask_treshold (int): Threshold value for mask.
        experiment (Experiment): Experiment object for parameter management and data collection.
        width_0 (int): Frame width.
        height_0 (int): Frame height.
        delta_x (int): Horizontal offset for region of interest.
        delta_y (int): Vertical offset for region of interest.
        x_start, x_end, y_start, y_end (int): Coordinates for region of interest.
        delay (float): Delay between frame processing iterations (in seconds).
    """

    changePixmap = pyqtSignal(QImage)

    def __init__(
        self,
        video_source,
        out_file_name,
        do_analize=False,
        draw_mask=False,
        mask_treshold=140,
        orientation=0,
    ):
        """
        Initialize the FrameThread.

        Args:
            source (str or int): Path to video file or camera index.
        """
        super().__init__()

        self.out_file_name = out_file_name
        self.video_source = video_source
        self.do_analize = do_analize
        self.draw_mask = draw_mask
        self.mask_treshold = mask_treshold
        self.orientation = (
            orientation  # 0 - no rotation, 1 - +90 degrees, 2 - -90 degrees
        )

        self.experiment = Experiment(out_file_name, source_name=video_source)
        self.running = True
        self.width_0 = 0
        self.height_0 = 0
        self.delta_x = 0
        self.delta_y = 0
        self.x_start, self.x_end = 0, 0
        self.y_start, self.y_end = 0, 0
        self.delay = 0.3

        logger.info("FrameThread initialized for source=%s, output=%s", video_source, out_file_name)

    def set_frame_size(self, frame):
        self.width_0 = frame.shape[1]
        self.height_0 = frame.shape[0]
        self.x_end = self.width_0
        self.y_end = self.height_0

    def run(self):
        """
        Main thread loop for capturing and processing video frames.

        - Captures frames from the video source.
        - Applies region of interest and optional masking.
        - If analysis is enabled, extracts angle and voltage data using the Experiment object.
        - Emits processed frames to the GUI via the changePixmap signal.
        - Saves measurement data and error analysis after processing is complete.
        """
        cap = cv2.VideoCapture(self.video_source)
        self.running = True

        while self.running:
            ret, frame = cap.read()
            time.sleep(self.delay)
            if not ret:
                cap.release()
                logger.info("Video source ended or cannot read frame.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.orientation == 1:  # +90 degrees
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.orientation == 2:  # -90 degrees
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            logger.debug(f"Processing frame: {frame.shape}")
            self.set_frame_size(frame)
            ch = frame.shape[2]
            bytes_per_line = ch * self.width_0
            frame_crop = frame[
                self.y_start + self.delta_y : self.y_end + self.delta_y,
                self.x_start + self.delta_x : self.x_end + self.delta_x,
            ]
            frame_crop = cv2.resize(frame_crop, (self.width_0, self.height_0))

            if self.draw_mask:
                # color white mask
                frame_crop = frame_crop[:, :, 2]
                frame_crop = (frame_crop > self.mask_treshold).astype(np.uint8) * 255
                frame_crop = cv2.cvtColor(frame_crop, cv2.COLOR_GRAY2RGB)

            if self.do_analize:
                p1, p2, phi = self.experiment.extract_angle(frame_crop, self.mask_treshold)
                if 60 < phi < 90:
                    self.experiment.predict_label(frame)
                cv2.line(frame_crop, p1, p2, (255, 0, 0), 5)

            image = QImage(frame_crop.data, self.width_0, self.height_0, bytes_per_line, 13)
            image = image.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(image)

        self.experiment.serialize()
        logger.info("FrameThread finished processing and serialized results.")

    def stop(self):
        """
        Stop the frame processing thread.
        """
        self.running = False
