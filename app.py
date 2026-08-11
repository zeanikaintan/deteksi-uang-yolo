import streamlit as st
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="Deteksi Uang", layout="wide")

st.title("💵 Deteksi Uang Realtime dengan YOLOv8")

# Load model
model = YOLO("best.pt")

# Tombol start kamera
run = st.checkbox("Aktifkan Kamera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

# ==========================
# KALIBRASI SEMENTARA
# ==========================
KNOWN_WIDTH = 15.0      # cm
FOCAL_LENGTH = 700      # Sesuaikan jika diperlukan

while run:
    success, frame = camera.read()

    if not success:
        st.error("Webcam tidak ditemukan")
        break

    # Deteksi YOLO
    results = model(frame)

    # Salin frame
    annotated_frame = frame.copy()

    # Loop semua objek yang terdeteksi
    for box in results[0].boxes:

        # Koordinat bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Nama kelas
        cls = int(box.cls[0])
        label = model.names[cls]

        # Lebar bounding box
        box_width = x2 - x1

        if box_width <= 0:
            continue

        # Estimasi jarak
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width

        # Batasi maksimal 35 cm
        if distance > 35:
            continue

        # Bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label + Jarak
        cv2.putText(
            annotated_frame,
            f"{label} | {int(distance)} cm",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Tampilkan hasil
    FRAME_WINDOW.image(
        annotated_frame,
        channels="BGR"
    )

camera.release()