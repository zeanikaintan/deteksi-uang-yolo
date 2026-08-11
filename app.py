import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av
import cv2


# =========================
# PENGATURAN HALAMAN
# =========================

st.set_page_config(
    page_title="Deteksi Uang",
    layout="wide"
)

st.title("💵 Deteksi Uang Realtime dengan YOLOv8")
st.write("Arahkan kamera ke uang Rupiah untuk melakukan deteksi.")


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================
# KALIBRASI SEMENTARA
# =========================

KNOWN_WIDTH = 15.0
FOCAL_LENGTH = 700


# =========================
# PROSES VIDEO
# =========================

def video_frame_callback(frame):

    # Ambil frame dari kamera browser
    img = frame.to_ndarray(format="bgr24")

    # Deteksi YOLO
    results = model(img, verbose=False)

    # Salin frame untuk diberi bounding box
    annotated_frame = img.copy()

    # Semua objek yang terdeteksi
    for box in results[0].boxes:

        # Koordinat bounding box
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Class
        cls = int(box.cls[0])
        label = model.names[cls]

        # Confidence
        confidence = float(box.conf[0])

        # Lebar bounding box
        box_width = x2 - x1

        if box_width <= 0:
            continue

        # Estimasi jarak
        distance = (
            KNOWN_WIDTH * FOCAL_LENGTH
        ) / box_width

        # Batasi sementara maksimal 35 cm
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

        # Tulisan label
        text = (
            f"{label} | "
            f"{confidence * 100:.1f}% | "
            f"{distance:.1f} cm"
        )

        # Posisi tulisan
        text_y = max(y1 - 10, 25)

        cv2.putText(
            annotated_frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Kembalikan frame ke browser
    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


# =========================
# KAMERA BROWSER
# =========================

webrtc_streamer(
    key="deteksi-uang",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)
