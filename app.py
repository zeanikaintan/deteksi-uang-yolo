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
    page_icon="💵",
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
# KALIBRASI JARAK
# =========================

KNOWN_WIDTH = 15.0
FOCAL_LENGTH = 700


# =========================
# PROSES FRAME VIDEO
# =========================

def video_frame_callback(frame):

    # Ambil frame dari kamera browser
    img = frame.to_ndarray(format="bgr24")

    # =========================
    # DETEKSI YOLO
    # =========================

    results = model.predict(
        source=img,
        conf=0.15,
        imgsz=640,
        verbose=False
    )

    # Salin frame
    annotated_frame = img.copy()


    # =========================
    # LOOP SEMUA OBJEK
    # =========================

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


        # =========================
        # ESTIMASI JARAK
        # =========================

        distance = (
            KNOWN_WIDTH * FOCAL_LENGTH
        ) / box_width


        # Batasi maksimal 35 cm
        if distance > 35:
            continue


        # =========================
        # BOUNDING BOX
        # =========================

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # =========================
        # LABEL
        # =========================

        text = (
            f"{label} | "
            f"{confidence * 100:.1f}% | "
            f"{distance:.1f} cm"
        )

        cv2.putText(
            annotated_frame,
            text,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # =========================
    # KEMBALIKAN FRAME
    # =========================

    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


# =========================
# WEBRTC CAMERA
# =========================

webrtc_streamer(
    key="deteksi-uang",
    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    # Pemrosesan video secara asynchronous
    async_processing=True,

    # =========================
    # STUN + TURN
    # =========================

    rtc_configuration={
        "iceServers": [

            # STUN Google
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            },

            # TURN Open Relay
            {
                "urls": [
                    "turn:openrelay.metered.ca:80"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },

            # TURN melalui port 443
            {
                "urls": [
                    "turn:openrelay.metered.ca:443"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },

            # TURN TCP
            {
                "urls": [
                    "turn:openrelay.metered.ca:443?transport=tcp"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    },

    # Sembunyikan kontrol audio karena tidak digunakan
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
