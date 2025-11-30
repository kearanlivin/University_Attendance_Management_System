import os
import io
import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration & Model Paths ---
MODEL_FILENAME = r"D:\User\USER.ME\MEAN PISETH\UAS\Model\final_mobilenetv2_head_model.h5"
DLIB_PREDICTOR_FILENAME = r"D:\User\USER.ME\MEAN PISETH\UAS\shape_predictor_68_face_landmarks.dat"
RECOG_CONF_THRESHOLD = 0.5  # cosine similarity threshold

# --- Dynamic Imports for Error Tolerance ---
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
st.set_page_config(layout="wide", page_title="UAS - Facial Landmark Detector & Auto-Register")
st.title("🎓 University Attendance Management System - UAS (Auto-Register Demo)")
st.subheader("Face Detection, 68-Point Landmark Extraction, Auto-registration with in-memory DB")

# --- Model Loading ---
@st.cache_resource
def load_models():
    local_model = None
    local_dlib_predictor = None

    # Load MobileNetV2 Model
    if tf is not None and os.path.exists(MODEL_FILENAME):
        try:
            local_model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
            logger.info("MobileNetV2 model loaded.")
        except Exception as e:
            logger.error(f"Model load error: {e}")

    # If .dat missing but .dat.bz2 exists, attempt to decompress automatically
    bz2_path = DLIB_PREDICTOR_FILENAME + ".bz2"
    if dlib is not None and not os.path.exists(DLIB_PREDICTOR_FILENAME) and os.path.exists(bz2_path):
        try:
            import bz2
            logger.info(f"Found {bz2_path}, attempting to decompress to {DLIB_PREDICTOR_FILENAME}")
            with open(bz2_path, "rb") as f_in:
                data = bz2.decompress(f_in.read())
            with open(DLIB_PREDICTOR_FILENAME, "wb") as f_out:
                f_out.write(data)
            logger.info("Decompression successful.")
        except Exception as e:
            logger.error(f"Failed to decompress {bz2_path}: {e}")
            # surface to user as well
            st.warning(f"Failed to decompress {bz2_path}: {e}")

    # Load Dlib Predictor
    if dlib is not None and os.path.exists(DLIB_PREDICTOR_FILENAME):
        try:
            local_dlib_predictor = dlib.shape_predictor(DLIB_PREDICTOR_FILENAME)
            logger.info("dlib predictor loaded.")
        except Exception as e:
            logger.error(f"Dlib load error: {e}")
            st.warning(f"Dlib load error: {e}")
    else:
        if dlib is None:
            logger.warning("dlib package not installed; landmarks will be unavailable.")
        else:
            logger.warning(f"{DLIB_PREDICTOR_FILENAME} not found; landmarks will be unavailable.")

    # OpenCV Haar cascade fallback
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade, local_dlib_predictor, local_model
# ...existing code...

face_cascade, dlib_predictor, model = load_models()
# quick runtime debug visible in Streamlit and logs
logger.info(f"dlib_predictor loaded: {bool(dlib_predictor)} path={DLIB_PREDICTOR_FILENAME}")
st.write("dlib_predictor loaded:", bool(dlib_predictor))

# show predictor status in UI so it's obvious why landmarks are empty
if dlib_predictor is None:
    st.warning(
        "Dlib shape predictor not loaded — 68-point landmarks will be unavailable.\n"
        f"Expected file: {DLIB_PREDICTOR_FILENAME} (put it in app folder or set full path).\n"
        "Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    )
else:
    st.success("Dlib shape predictor loaded — 68-point landmarks enabled.")

# --- Temporary in-memory DB stored in session_state ---
def init_db():
    if 'uas_db' not in st.session_state:
        st.session_state['uas_db'] = {
            "users": [],      # list of dicts with user_id, name, embedding (np.array), landmarks (list or None), bbox (tuple)
            "next_idx": 1
        }

def safe_extract_landmarks(bgr_img, bbox, user_rec=None, draw_on_img=None):
    """
    Robust wrapper for 68-point landmark extraction.
    - bgr_img: OpenCV BGR image
    - bbox: (x, y, w, h)
    - user_rec: dict from DB, if provided, auto-updates landmarks
    - draw_on_img: optional image to draw landmarks
    Returns: list of 68 (x,y) points (never empty)
    """
    coords = extract_landmarks_for_bbox(bgr_img, bbox, draw_on_img=draw_on_img)
    
    if coords is None or len(coords) != 68:
        # fallback: zeroed landmarks if extraction fails
        coords = [(0, 0) for _ in range(68)]
        logger.warning(f"Landmark extraction failed for bbox {bbox}, returning zeros")
    
    # convert np array -> list if needed
    if isinstance(coords, np.ndarray):
        coords = coords.tolist()
    
    # Update DB automatically if user_rec provided
    if user_rec is not None:
        update_user_landmarks(user_rec, coords)
    
    return coords


def generate_fake_name(idx):
    return f"User_{idx:03d}"

def register_user(embedding, landmarks, bbox):
    """
    Create new user entry in in-memory DB and return stored record.
    Only store landmarks if they are valid 68-point list.
    """
    db = st.session_state['uas_db']
    idx = db['next_idx']
    user_id = f"user_{idx:03d}"
    name = generate_fake_name(idx)
    record = {
        "user_id": user_id,
        "name": name,
        "embedding": embedding.astype(np.float32) if isinstance(embedding, np.ndarray) else np.array(embedding, dtype=np.float32),
        # store landmarks only if we have 68 points; otherwise store None
        "landmarks": landmarks if (isinstance(landmarks, list) and len(landmarks) == 68) else None,
        "bbox": bbox
    }
    db['users'].append(record)
    db['next_idx'] += 1
    logger.info(f"Registered new user: {user_id} ({name}) - landmarks stored: {'yes' if record['landmarks'] else 'no'}")
    return record

def update_user_landmarks(user_rec, landmarks):
    """
    Update stored landmarks for an existing user only if landmarks is a 68-point list.
    """
    if isinstance(landmarks, list) and len(landmarks) == 68:
        user_rec['landmarks'] = landmarks
        logger.info(f"Updated landmarks for user {user_rec['user_id']}")
        return True
    return False

def cosine_similarity(a, b):
    """
    Compute cosine similarity between 1D numpy arrays.
    Returns float in [-1,1]. Handles zero vectors.
    """
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def match_embedding(embedding, threshold=RECOG_CONF_THRESHOLD):
    """
    Match embedding against DB. Returns (best_record_or_None, best_score).
    Uses only embeddings for matching (never landmarks).
    """
    db = st.session_state['uas_db']
    best_score = -1.0
    best_rec = None
    for rec in db['users']:
        score = cosine_similarity(embedding, rec['embedding'])
        if score > best_score:
            best_score = score
            best_rec = rec
    if best_rec is not None and best_score >= threshold:
        return best_rec, best_score
    return None, best_score

# --- Image/Detection Helpers ---
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

def detect_faces_only(bgr_img, draw=True):
    """
    Detect faces with OpenCV cascade but DO NOT extract dlib landmarks here.
    Returns image (with bboxes drawn if draw=True) and list of dicts: {"bbox":(x,y,w,h), "face_crop":...}
    """
    if bgr_img is None:
        return bgr_img, []
    img = bgr_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    results = []
    for (x, y, w, h) in faces:
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
        face_crop = img[y1:y2, x1:x2].copy()
        if draw:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        results.append({
            "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            "face_crop": face_crop
        })
    return img, results

# ...existing code...
def extract_landmarks_for_bbox(bgr_img, bbox, draw_on_img=None):
    """
    Robust dlib 68-point extractor:
    - ensures uint8 contiguous RGB/gray image for dlib
    - clamps bbox to image
    - tries predictor on full image, then on face-crop fallback
    - draws landmarks onto draw_on_img if provided
    """
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    x1c = max(0, x)
    y1c = max(0, y)
    x2c = min(bgr_img.shape[1]-1, x2)
    y2c = min(bgr_img.shape[0]-1, y2)

    if dlib_predictor is None:
        logger.debug("extract_landmarks_for_bbox: dlib_predictor is None")
        return None

    if bgr_img is None:
        return None

    # Ensure uint8 and contiguous
    img = bgr_img
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Convert BGR/BGRA/GRAY -> RGB (dlib accepts RGB or gray)
    try:
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img.copy()
        img_rgb = np.ascontiguousarray(img_rgb)
    except Exception as e:
        logger.exception(f"convert to RGB failed: {e}")
        return None

    h_img, w_img = img_rgb.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w_img - 1, x2), min(h_img - 1, y2)
    if x2c <= x1c or y2c <= y1c:
        logger.warning(f"Invalid/clamped bbox: {(x1,y1,x2,y2)} -> {(x1c,y1c,x2c,y2c)} img_shape={img_rgb.shape}")
        return None

    # Try predictor on full RGB image
    try:
        rect = dlib.rectangle(int(x1c), int(y1c), int(x2c), int(y2c))
        shape = dlib_predictor(img_rgb, rect)
        coords = [(int(shape.part(i).x), int(shape.part(i).y)) for i in range(68)]
        if draw_on_img is not None and coords:
            for lx, ly in coords:
                cv2.circle(draw_on_img, (lx, ly), 2, (0, 255, 0), -1)
        if len(coords) == 68:
            return coords
    except Exception as e:
        logger.exception(f"dlib extraction on full image failed: {e}")

    # Fallback: predictor on the face crop (map coordinates back)
    try:
        crop = img_rgb[y1c:y2c + 1, x1c:x2c + 1]
        if crop.size == 0:
            logger.warning("face crop empty in fallback")
            return None
        crop = np.ascontiguousarray(crop)
        rect2 = dlib.rectangle(0, 0, crop.shape[1] - 1, crop.shape[0] - 1)
        shape2 = dlib_predictor(crop, rect2)
        coords2 = [(int(shape2.part(i).x) + x1c, int(shape2.part(i).y) + y1c) for i in range(68)]
        if draw_on_img is not None and coords2:
            for lx, ly in coords2:
                cv2.circle(draw_on_img, (lx, ly), 2, (0, 255, 0), -1)
        return coords2 if len(coords2) == 68 else None
    except Exception as e:
        logger.exception(f"dlib extraction on crop failed: {e}")
        return None
# ...existing code...

def get_embedding_from_face(face_crop):
    """
    Returns 1D numpy array embedding.
    If TF model available, use it. Otherwise fallback to flattened landmarks-length vector (zeros if not available).
    """
    if face_crop is None or face_crop.size == 0:
        return np.zeros((1,), dtype=np.float32)
    # Prefer model if available
    if model is not None:
        try:
            H, W = 224, 224
            face_resized = cv2.resize(face_crop, (W, H))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            x_input = np.expand_dims(face_rgb.astype("float32") / 255.0, axis=0)
            pred = model.predict(x_input, verbose=0)
            # ensure 1D vector
            emb = np.ravel(pred).astype(np.float32)
            logger.debug(f"Embedding (model) shape: {emb.shape}")
            return emb
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
    # Fallback: use simple embedding derived from resized gray pixels (deterministic)
    try:
        small = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        emb = small.flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        logger.debug("Using fallback pixel-based embedding")
        return emb
    except Exception as e:
        logger.error(f"Fallback embedding error: {e}")
        return np.zeros((1,), dtype=np.float32)

# --- Initialize DB ---
init_db()

# --- Redesigned UI: single sidebar with info + mode selection ---
with st.sidebar:
    st.header("UAS - Sidebar")
    st.markdown(
        """
        - Upload Photo or Webcam (Real-time).
        - Dlib 68-landmarks always extracted from the current frame (if dlib predictor available).
        - Recognition uses cosine similarity on embeddings (threshold 0.5).
        - New users auto-registered into an in-memory DB (persist for session).
        """
    )
    mode = st.selectbox("Select mode", ("Upload Photo", "Webcam (Real-time)"))
    st.markdown("---")
    st.caption("In-memory DB size: " + str(len(st.session_state['uas_db']['users'])))
    st.markdown("---")
    st.caption("UAS - Face Detection & Auto-registration")

# --- Processing helpers for UI flows ---
def process_and_match_image(bgr):
    """
    Flow (per user's rules):
    - detect -> embed -> match user -> extract landmarks -> return them in output
    - DB stores landmarks only if we extracted 68 points
    - Always return newly extracted landmarks (never DB landmarks)
    """
    annotated, detections = detect_faces_only(bgr.copy(), draw=True)
    output_results = []

    for det in detections:
        bbox = det["bbox"]
        face_crop = det["face_crop"]
        # 1) embed
        emb = get_embedding_from_face(face_crop)
        # debug embedding
        try:
            logger.debug(f"Embedding first8: {emb.flatten()[:8]}")
            st.write("Debug embedding (first 8):", emb.flatten()[:8].tolist())
        except Exception:
            logger.debug("Embedding debug suppressed (streamlit call failed)")

        # 2) match using only embeddings
        matched_rec, score = match_embedding(emb, threshold=RECOG_CONF_THRESHOLD)
        # debug similarity score
        logger.debug(f"Similarity score: {score}")
        try:
            st.write("Debug similarity score:", float(score))
        except Exception:
            pass

        # 3) extract fresh landmarks from current frame (always)
        landmarks = safe_extract_landmarks(annotated, bbox, user_rec=matched_rec if matched_rec else None, draw_on_img=annotated)
        
        # debug landmarks shape
        l_shape = len(landmarks) if isinstance(landmarks, list) else 0
        logger.debug(f"Extracted landmarks count: {l_shape}")
        try:
            st.write("Debug extracted landmarks count:", l_shape)
        except Exception:
            pass

        if matched_rec is not None:
            # recognized based on embedding only
            rec = {
                "recognized": True,
                "user_id": matched_rec["user_id"],
                "name": matched_rec["name"],
                "similarity_score": float(score),
                # MUST return newly extracted landmarks (not DB)
                "landmarks": landmarks if landmarks is not None else [],
                "bbox": bbox
            }
            # Update stored landmarks for DB only if we have full 68 points
            if landmarks and len(landmarks) == 68:
                update_user_landmarks(matched_rec, landmarks)
            output_results.append(rec)
        else:
            # register new user
            new_rec = register_user(emb, landmarks, bbox)
            rec = {
                "recognized": False,
                "user_id": new_rec["user_id"],
                "name": new_rec["name"],
                "similarity_score": float(score),
                # return newly extracted landmarks (never DB stored ones)
                "landmarks": landmarks if landmarks is not None else [],
                "bbox": bbox
            }
            output_results.append(rec)
    return annotated, output_results

# --- Main Content Area ---
if mode == "Upload Photo":
    st.header("1️⃣ Upload Image (Static Analysis & Auto-register)")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        bgr = to_cv2_image(image)
        processed, results = process_and_match_image(bgr)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                 caption=f"Detected {len(results)} faces (auto-registered if new)",
                 use_column_width=True)
        if results:
            st.markdown("### Results (one entry per detected face)")
            # Convert numpy types to lists for JSON display
            serializable = []
            for r in results:
                serializable.append({
                    "recognized": bool(r["recognized"]),
                    "user_id": r["user_id"],
                    "name": r["name"],
                    "similarity_score": float(r["similarity_score"]),
                    "landmarks": r["landmarks"],
                    "bbox": r["bbox"]
                })
            st.json(serializable)
        else:
            st.info("No faces detected in the uploaded image.")
    else:
        st.info("Upload an image to test face detection, landmark extraction, recognition and auto-registration.")

elif mode == "Webcam (Real-time)":
    st.header("2️⃣ Webcam (Real-time Detection & Auto-register)")

    if webrtc_streamer and VideoTransformerBase and av:
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        class LandmarkTransformer(VideoTransformerBase):
            def __init__(self):
                super().__init__()
                self.last_results = None

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                out, detections = detect_faces_only(img.copy(), draw=True)
                results = []
                for det in detections:
                    bbox = det["bbox"]
                    face_crop = det["face_crop"]
                    emb = get_embedding_from_face(face_crop)
                    # Avoid st calls inside transformer; use logger for debug
                    logger.debug(f"Transformer embedding first8: {emb.flatten()[:8] if emb is not None else None}")
                    matched_rec, score = match_embedding(emb, threshold=RECOG_CONF_THRESHOLD)
                    logger.debug(f"Transformer similarity score: {score}")

                    # Always extract fresh landmarks from the current frame
                    landmarks = extract_landmarks_for_bbox(out, bbox, draw_on_img=out)
                    l_shape = len(landmarks) if isinstance(landmarks, list) else 0
                    logger.debug(f"Transformer extracted landmarks count: {l_shape}")

                    if matched_rec is not None:
                        # Update stored landmarks only if full 68 points
                        if landmarks and len(landmarks) == 68:
                            update_user_landmarks(matched_rec, landmarks)
                        results.append({
                            "recognized": True,
                            "user_id": matched_rec["user_id"],
                            "name": matched_rec["name"],
                            "similarity_score": float(score),
                            "landmarks": landmarks if landmarks is not None else [],
                            "bbox": bbox
                        })
                    else:
                        new_rec = register_user(emb, landmarks, bbox)
                        results.append({
                            "recognized": False,
                            "user_id": new_rec["user_id"],
                            "name": new_rec["name"],
                            "similarity_score": float(score),
                            "landmarks": landmarks if landmarks is not None else [],
                            "bbox": bbox
                        })
                # store lightweight serializable results
                self.last_results = results
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

            try:
                while True:
                    transformer = getattr(webrtc_ctx, "video_transformer", None)
                    playing = getattr(webrtc_ctx.state, "playing", False)
                    if transformer is not None:
                        latest = getattr(transformer, "last_results", None)
                        if latest:
                            json_placeholder.json(latest)
                        else:
                            json_placeholder.info("No landmarks yet (waiting for frames).")
                    else:
                        json_placeholder.info("Video transformer not initialized. Start the stream.")
                    status_text = "Streaming" if playing else "Stopped"
                    status_placeholder.markdown(f"**Status:** {status_text}")
                    # Pause to allow UI interactions and avoid tight loop
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Realtime display loop error: {e}")
                st.info("Realtime display loop terminated. Interact with the page to restart or check logs.")
        else:
            # fallback camera capture if WebRTC failed
            cam_image = st.camera_input("Take a picture")
            if cam_image is not None:
                image = Image.open(cam_image).convert("RGB")
                bgr = to_cv2_image(image)
                processed, results = process_and_match_image(bgr)
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                         caption=f"Captured image: {len(results)} faces found",
                         use_column_width=True)
                if results:
                    st.markdown("### Extracted landmarks and recognition info")
                    st.json(results)
                else:
                    st.info("No faces detected in the captured image.")
    else:
        st.warning("WebRTC libraries are missing or incompatible. Falling back to single-shot camera capture.")
        cam_image = st.camera_input("Take a picture")
        if cam_image is not None:
            image = Image.open(cam_image).convert("RGB")
            bgr = to_cv2_image(image)
            processed, results = process_and_match_image(bgr)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                     caption=f"Captured image: {len(results)} faces found",
                     use_column_width=True)
            if results:
                st.markdown("### Extracted landmarks and recognition info")
                st.json(results)
            else:
                st.info("No faces detected in the captured image.")

st.markdown("---")
st.caption("✅ Status: In-memory DB persists while session active. Recognition threshold = 0.5. New users auto-registered.")