import os
import io
import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st
import logging

# Set up logging for dlib errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Configuration & Model Paths (unchanged) ---
MODEL_FILENAME = "mobilenetv2_head_best.h5"
DLIB_PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"
RECOG_CONF_THRESHOLD = 0.6  # only extract landmarks if recognition confidence >= this

# --- Dynamic Imports for Error Tolerance (unchanged) ---
tf = None
dlib = None
webrtc_streamer = None
VideoTransformerBase = None
RTCConfiguration = None
av = None

try:
    import tensorflow as tf
except Exception:
    pass

try:
    import dlib
except Exception:
    pass

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    import av
except Exception:
    pass

# --- Streamlit Setup ---
st.set_page_config(layout="wide", page_title="UAS - Facial Landmark Detector")
st.title("🎓 University Attendance Management System - UAS")
st.subheader("Face Detection, 68-Point Landmark Extraction, & MobileNetV2")

# --- Model Loading (unchanged) ---
@st.cache_resource
def load_models():
    local_model = None
    local_dlib_predictor = None

    # Load MobileNetV2 Model
    if tf is not None and os.path.exists(MODEL_FILENAME):
        try:
            local_model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
            st.success("MobileNetV2 Recognition Model loaded.")
        except Exception as e:
            st.error(f"FATAL Model Load Error (TensorFlow/NumPy Mismatch): {e}")

    # Load Dlib Predictor
    if dlib is not None and os.path.exists(DLIB_PREDICTOR_FILENAME):
        try:
            local_dlib_predictor = dlib.shape_predictor(DLIB_PREDICTOR_FILENAME)
            st.success("dlib 68-point predictor loaded.")
        except Exception as e:
            st.error(f"FATAL Dlib Predictor Error (NumPy Mismatch likely): {e}")

    # Load OpenCV's Haar Cascade (Fallback Face Detector)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    return face_cascade, local_dlib_predictor, local_model

face_cascade, dlib_predictor, model = load_models()

# --- Helper Functions (unchanged) ---
def to_cv2_image(pil_img):
    """Converts PIL RGB Image to OpenCV BGR numpy.uint8 array."""
    img = np.array(pil_img).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def detect_faces_and_landmarks(bgr_img, draw=True):
    """
    Detects faces, runs optional recognition first, and only then runs dlib landmarks
    if the face is recognized (confidence >= RECOG_CONF_THRESHOLD). Behavior unchanged.
    Returns annotated image and list of dicts: {"bbox":(x,y,w,h), "face_crop":..., "landmarks":...}
    """
    if bgr_img is None or bgr_img.size == 0:
        return bgr_img, []

    bgr_img = bgr_img.copy()
    if bgr_img.dtype != np.uint8:
        bgr_img = bgr_img.astype(np.uint8)

    img_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    results = []

    for (x, y, w, h) in faces:
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, bgr_img.shape[1]), min(y + h, bgr_img.shape[0])
        face_crop = bgr_img[y1:y2, x1:x2].copy()
        landmarks = None
        label = "Face Detected"
        recognized = False

        # 1) Recognition first (if model available)
        if model is not None and face_crop.size != 0:
            try:
                H, W = 224, 224
                face_resized = cv2.resize(face_crop, (W, H))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                x_input = np.expand_dims(face_rgb.astype("float32") / 255.0, axis=0)
                pred = model.predict(x_input, verbose=0)
                if pred.ndim == 2 and pred.shape[1] > 1:
                    class_id = int(np.argmax(pred[0]))
                    conf = float(pred[0][class_id])
                    label = f"Student ID: {class_id} ({conf:.2f})"
                    if conf >= RECOG_CONF_THRESHOLD:
                        recognized = True
                else:
                    label = "Recognition Model Output"
            except Exception as e:
                logger.error(f"Recognition error: {e}")
                label = "Recog Error"

        # 2) Only run dlib landmarks if face recognized (or if no model is present)
        if dlib_predictor is not None:
            run_dlib = False
            if model is None:
                run_dlib = True
            else:
                run_dlib = recognized

            if run_dlib:
                try:
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    shape = dlib_predictor(img_gray, rect)
                    coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                    landmarks = coords
                    if draw:
                        for (lx, ly) in coords:
                            cv2.circle(bgr_img, (lx, ly), 2, (0, 255, 0), -1)
                except Exception as e:
                    logger.error(f"dlib predictor error: {e}")
                    landmarks = None

        # 3) Draw bounding box and label
        if draw:
            cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(bgr_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        results.append({"bbox": (x, y, w, h), "face_crop": face_crop, "landmarks": landmarks, "recognized": recognized})

    return bgr_img, results
    
# --- Redesigned UI: single sidebar with info + mode selection ---
with st.sidebar:
    st.header("UAS - Sidebar")
    st.markdown(
        """
        This sidebar contains app info and controls.
        - Choose detection mode: Upload Photo or Webcam (Real-time).
        - The recognition model (MobileNetV2) runs first. Dlib 68-landmarks run only when recognition is confident (or when no model present).
        """
    )
    mode = st.selectbox("Select mode", ("Upload Photo", "Webcam (Real-time)"))
    st.markdown("---")
    st.caption("Model and dlib loads are attempted on app start. See main page for load status.")
    st.markdown("---")
    st.caption("UAS - Face Detection & Landmark Extraction")

# --- Main Content Area ---
if mode == "Upload Photo":
    st.header("1️⃣ Upload Image (Static Analysis)")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        bgr = to_cv2_image(image)
        processed, results = detect_faces_and_landmarks(bgr.copy(), draw=True)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                 caption=f"Detected {len(results)} faces (landmarks extracted only when recognized/confident)",
                 use_column_width=True)

        # --- NEW: Print/Display extracted landmarks and recognition info ---
        display_results = []
        for i, r in enumerate(results):
            display_results.append({
                "face_index": i + 1,
                "bbox": r["bbox"],
                "recognized": r["recognized"],
                "landmarks_count": len(r["landmarks"]) if r["landmarks"] else 0,
                "landmarks": r["landmarks"],  # can be None or list of (x,y)
            })
        if display_results:
            st.markdown("### Extracted landmarks and recognition info")
            st.json(display_results)
        else:
            st.info("No faces detected in the uploaded image.")
    else:
        st.info("Upload an image to test face detection, landmark extraction, and recognition.")

elif mode == "Webcam (Real-time)":
    st.header("2️⃣ Webcam (Real-time Detection)")

    # If WebRTC libs are available, use them; otherwise fallback to single-shot camera input
    if webrtc_streamer and VideoTransformerBase and av:
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class LandmarkTransformer(VideoTransformerBase):
            def __init__(self):
                super().__init__()
                self.last_results = None

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                out, results = detect_faces_and_landmarks(img, draw=True)
                # store only serializable pieces (avoid raw face_crop numpy arrays)
                serializable = []
                for r in results:
                    serializable.append({
                        "bbox": r["bbox"],
                        "recognized": r["recognized"],
                        "landmarks": r["landmarks"],
                    })
                self.last_results = serializable
                return av.VideoFrame.from_ndarray(out, format="bgr24")

        try:
            st.success("WebRTC stream is available. Click Start below.")
            webrtc_ctx = webrtc_streamer(
                key="uas-webcam",
                rtc_configuration=RTC_CONFIGURATION,
                video_transformer_factory=LandmarkTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        except Exception as e:
            logger.error(f"webrtc_streamer failed: {e}")
            st.warning("WebRTC failed or is incompatible in this environment. Falling back to single-shot camera capture.")
            webrtc_ctx = None

        # Real-time JSON display updated while stream is running
        if 'webrtc_ctx' in locals() and webrtc_ctx is not None:
            info_col, json_col = st.columns([1, 1])
            with info_col:
                st.markdown("### Stream Status")
                status_placeholder = st.empty()
                st.markdown("### Controls")
                st.caption("Use the Start/Stop button created by the WebRTC widget to control the stream.")
            with json_col:
                st.markdown("### Latest frame landmarks (live)")
                json_placeholder = st.empty()

            # Poll the transformer for latest results while the stream is active
            try:
                while True:
                    # when webrtc widget hasn't been started, video_transformer may be None
                    transformer = getattr(webrtc_ctx, "video_transformer", None)
                    playing = getattr(webrtc_ctx.state, "playing", False)
                    if transformer is not None:
                        latest = getattr(transformer, "last_results", None)
                        if latest:
                            json_placeholder.json(latest)
                        else:
                            json_placeholder.info("No landmarks yet (waiting for a recognized face/frame).")
                    else:
                        json_placeholder.info("Video transformer not initialized. Start the stream.")
                    # update status
                    status_text = "Streaming" if playing else "Stopped"
                    status_placeholder.markdown(f"**Status:** {status_text}")
                    # break the loop when the app reruns or stream is stopped to avoid locking UI indefinitely
                    if not playing:
                        # sleep a bit to allow user to start the stream
                        time.sleep(0.5)
                    else:
                        time.sleep(0.5)
                    # Streamlit reruns on user interaction; this loop will exit when the script is re-executed by Streamlit.
            except Exception as e:
                logger.error(f"Realtime display loop error: {e}")
                st.info("Realtime display loop terminated. Interact with the page to restart or check logs.")
        else:
            # fallback camera capture if WebRTC failed
            cam_image = st.camera_input("Take a picture")
            if cam_image is not None:
                image = Image.open(cam_image).convert("RGB")
                bgr = to_cv2_image(image)
                processed, results = detect_faces_and_landmarks(bgr.copy(), draw=True)
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                         caption=f"Captured image: {len(results)} faces found",
                         use_column_width=True)
                display_results = []
                for i, r in enumerate(results):
                    display_results.append({
                        "face_index": i + 1,
                        "bbox": r["bbox"],
                        "recognized": r["recognized"],
                        "landmarks_count": len(r["landmarks"]) if r["landmarks"] else 0,
                        "landmarks": r["landmarks"],
                    })
                if display_results:
                    st.markdown("### Extracted landmarks and recognition info")
                    st.json(display_results)
                else:
                    st.info("No faces detected in the captured image.")
    else:
        st.warning("WebRTC libraries are missing or incompatible. Falling back to single-shot camera capture.")
        cam_image = st.camera_input("Take a picture")
        if cam_image is not None:
            image = Image.open(cam_image).convert("RGB")
            bgr = to_cv2_image(image)
            processed, results = detect_faces_and_landmarks(bgr.copy(), draw=True)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                     caption=f"Captured image: {len(results)} faces found",
                     use_column_width=True)
            display_results = []
            for i, r in enumerate(results):
                display_results.append({
                    "face_index": i + 1,
                    "bbox": r["bbox"],
                    "recognized": r["recognized"],
                    "landmarks_count": len(r["landmarks"]) if r["landmarks"] else 0,
                    "landmarks": r["landmarks"],
                })
            if display_results:
                st.markdown("### Extracted landmarks and recognition info")
                st.json(display_results)
            else:
                st.info("No faces detected in the captured image.")

st.markdown("---")
st.caption("✅ Status: UI updated. Model behavior unchanged.")