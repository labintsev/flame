# Flame Analyzer

[Документация на русском языке](readme.ru.md)

Flame Analyzer is a PyQt5-based application for analyzing video or camera streams to measure and process experimental data, such as angles and termo voltages, with error analysis and reporting.

## Features

- Real-time video/camera frame processing using OpenCV and PyQt5.
- Masking and thresholding for feature detection.
- Automatic calculation of angles and voltages from video frames.
- Error analysis: arithmetic mean, absolute and relative errors, total error, and confidence intervals.
- Saves measurement data to CSV or Excel, with detailed report generation.
- Interactive GUI for video selection, mask sensitivity, and region of interest adjustments.

## Requirements

- Python 3.7+
- PyQt5
- OpenCV (`opencv-python`)
- NumPy
- pandas

Install dependencies:

```sh
pip install -r requirements.txt
```

Serialize conda env:
```
conda pack -n .conda -o env.tar.gz
```

## Usage
Run the application:
```
python main.py
```

Select a video file or camera source.
Adjust mask sensitivity and region of interest as needed.
Start viewing or analyzing the video.
Save results and reports after analysis.

## File Structure
gui.py - Main application and GUI logic.
ui_1.py - PyQt5 UI definition (generated from Qt Designer).
readme.md - Project documentation.

## Output
Measurement data saved as .csv 
Error analysis and statistics saved as a .txt report.

License: MIT License
Authors: Andrej, Max, Pasha 
