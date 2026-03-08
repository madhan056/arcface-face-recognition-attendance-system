# ArcFace Face Recognition Attendance System

## Project Overview

This project is a **real-time attendance management system** that uses
**facial recognition technology** to automatically record attendance.\
The system detects and verifies faces using the **ArcFace deep learning
model** from the **InsightFace library** and records attendance in a
**MySQL database**.

The application captures images from a **live camera feed**, compares
detected faces with **stored facial embeddings**, and marks attendance
when a match is found.\
This provides a **contactless and automated attendance solution** that
improves **accuracy and efficiency**.

# Technologies Used

## Programming & Libraries
-   **Python**
-   **OpenCV (cv2)**
-   **NumPy**
-   **InsightFace**
-   **Face Recognition**

## Model
-   **ArcFace Model**
-   **InsightFace FaceAnalysis**

## Database
-   **MySQL**

## Utilities
-   **Pickle**
-   **Logging**

# Installation & Setup

## 1. Clone the Repository

``` bash
git clone https://github.com/madhan056/arcface-face-recognition-attendance-system.git
cd arcface-face-recognition-attendance-system
```

## 2. System Requirements

Before installing Python dependencies, install **Visual Studio Build
Tools**.

Run the following command in **PowerShell**:

``` powershell
winget install -e --id Microsoft.VisualStudio.2022.BuildTools
```

## 3. Model Setup

Navigate to the InsightFace directory:  .insightface/

Create a folder named:  model

Download the **ArcFace buffalo_l model** and move it into the **model**
folder.

## 4. Install Python Dependencies

Navigate to the project directory and install required libraries:

``` bash
pip install -r requirements.txt
```

# Running the Application

Run the main program:

``` bash
python main.py
```

The system will:
-   Open the **camera feed**
-   Detect faces in **real-time**
-   Compare **facial embeddings**
-   Mark **attendance automatically**
-   Store **attendance records in the MySQL database**

# Modules Used

## File Handling
-   `os` -- File and directory operations\
-   `pickle` -- Saving and loading facial embeddings

## Image Processing
-   `cv2` -- Image capture and processing\
-   `numpy` -- Numerical operations for image data

## Face Recognition
-   `insightface` -- Deep learning face recognition library\
-   `FaceAnalysis` -- Face detection and feature extraction

## Database & Time
-   `mysql.connector` -- Connect and interact with MySQL database\
-   `datetime` -- Handle attendance timestamps