import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av
import cv2


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Deteksi Uang Rupiah",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    color: #cbd5e1;
}

.card {
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    background-color: #ffffff;
    margin-bottom: 15px;
}

.card h3 {
    margin-top: 0;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================================================
# KALIBRASI JARAK
# =========================================================

KNOWN_WIDTH = 15.0
FOCAL_LENGTH = 700


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Beranda"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2>💵 Deteksi Uang</h2>
        <p style="color:gray;">
        Sistem Deteksi Nominal Uang Rupiah
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Menu")

    if st.button(
        "🏠  Beranda",
        use_container_width=True
    ):
        st.session_state.page = "Beranda"

    if st.button(
        "📷  Deteksi Real-Time",
        use_container_width=True
    ):
        st.session_state.page = "Deteksi"


# =========================================================
# HALAMAN BERANDA
# =========================================================

if st.session_state.page == "Beranda":

    st.markdown(
        """
        <div class="hero">

        <h1>💵 Deteksi Uang Rupiah Real-Time</h1>

        <p>
        Sistem deteksi nominal uang Rupiah menggunakan
        YOLOv8 dan teknologi WebRTC.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>📷 Deteksi Real-Time</h3>

            <p>
            Menggunakan kamera perangkat untuk mendeteksi
            uang Rupiah secara langsung.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🤖 YOLOv8</h3>

            <p>
            Model computer vision digunakan untuk
            mengenali nominal uang berdasarkan citra kamera.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("### ✨ Fitur Utama")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="info-box">

            <h4>💵 Nominal</h4>

            <p>
            Mengenali berbagai nominal uang Rupiah.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="info-box">

            <h4>🎯 Confidence</h4>

            <p>
            Menampilkan tingkat keyakinan hasil deteksi.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="info-box">

            <h4>📏 Estimasi Jarak</h4>

            <p>
            Menampilkan perkiraan jarak objek dari kamera.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")

    st.info(
        "Gunakan menu **Deteksi Real-Time** di sebelah kiri "
        "untuk mulai menggunakan kamera."
    )


# =========================================================
# FUNGSI PROSES VIDEO
# =========================================================

def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")


    # YOLO

    results = model.predict(
        source=img,
        conf=0.15,
        imgsz=640,
        verbose=False
    )


    annotated_frame = img.copy()


    # Semua objek

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        cls = int(box.cls[0])

        label = model.names[cls]

        confidence = float(box.conf[0])


        box_width = x2 - x1


        if box_width <= 0:
            continue


        # Jarak

        distance = (
            KNOWN_WIDTH * FOCAL_LENGTH
        ) / box_width


        # Maksimal 35 cm

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


        # Label

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


    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


# =========================================================
# HALAMAN DETEKSI REAL-TIME
# =========================================================

if st.session_state.page == "Deteksi":

    st.markdown(
        """
        <div class="hero">

        <h1>📷 Deteksi Real-Time</h1>

        <p>
        Arahkan kamera ke uang Rupiah untuk melakukan
        deteksi nominal secara langsung.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### 🎥 Kamera")


    # =====================================================
    # WEBRTC
    # =====================================================

    webrtc_streamer(

        key="deteksi-uang",

        video_frame_callback=video_frame_callback,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True,

        rtc_configuration={

            "iceServers": [

                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                },

                {
                    "urls": [
                        "turn:openrelay.metered.ca:80"
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                },

                {
                    "urls": [
                        "turn:openrelay.metered.ca:443"
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                },

                {
                    "urls": [
                        "turn:openrelay.metered.ca:443?transport=tcp"
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }

            ]

        }

    )


    st.markdown("---")


    # =====================================================
    # INFORMASI
    # =====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            "YOLOv8"
        )


    with col2:

        st.metric(
            "Mode",
            "Real-Time"
        )


    with col3:

        st.metric(
            "Jarak Maksimum",
            "35 cm"
        )


    st.info(
        "💡 Tips: Posisikan uang di depan kamera dengan "
        "pencahayaan yang cukup agar deteksi lebih optimal."
    )
