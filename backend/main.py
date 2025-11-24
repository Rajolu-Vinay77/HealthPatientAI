import cv2
import time
import numpy as np
import mediapipe as mp
import threading
import queue
import json
import os
import asyncio
from collections import deque, Counter
from datetime import datetime

# FastAPI Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ML Imports
from vosk import Model, KaldiRecognizer
import pyaudio
from deepface import DeepFace
import google.generativeai as genai
from transformers import pipeline

# --- REPLACEMENT: HSEmotion for Fast/Accurate Face Analysis ---
try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEmotion_AVAILABLE = True
except ImportError:
    print("[WARN] HSEmotion not found. Using DeepFace fallback.")
    HSEmotion_AVAILABLE = False

import librosa

# ---------------- CONFIGURATION ----------------

GEMINI_API_KEY = "AIzaSyCrLslzmvtWo969f7-ERdoLCXZ-dw61mS8" 

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_name = 'models/gemini-2.5-flash'
        gemini_model = genai.GenerativeModel(model_name)
        print(f"[INFO] Gemini API initialized: {model_name}")
    except Exception as e:
        print(f"[ERROR] Gemini Config Failed: {e}")

try:
    vocal_pipeline = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    print("[INFO] Vocal Emotion Model Loaded")
except:
    vocal_pipeline = None
    print("[WARN] Transformers library missing. Vocal analysis disabled.")

# Load the Fast Emotion Model
fer = None
if HSEmotion_AVAILABLE:
    try:
        fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        print("[INFO] HSEmotion Model Loaded (Fast Mode)")
    except Exception as e:
        print(f"[ERROR] HSEmotion Failed: {e}")

# ---------------- CONSTANTS ----------------
LOOK_THRESHOLD_X = 0.22
LOOK_THRESHOLD_Y = 0.20
HEAD_YAW_THRESHOLD = 25.0
HEAD_PITCH_THRESHOLD = 25.0
EAR_THRESHOLD = 0.12
GAZE_SMOOTHING = 6

# ---------------- SHARED STATE ----------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "status": "System Ready",
            "face_emotion": "Neutral",
            "face_conf": 0.0,
            "gaze_status": "N/A",
            "blink_state": "Open",
            "vocal_emotion": "N/A",
            "text": "",
            "sentiment": "Neutral",
            "clinical_flag": "None",
            "alert_active": False
        }
        self.text_alert_expires = 0.0
        
        # Frame buffer
        self.latest_frame = None 
        self.new_frame_available = False

    def update(self, key, value):
        with self.lock:
            self.data[key] = value
            
    def set_frame_for_analysis(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            self.new_frame_available = True

    def get_frame_for_analysis(self):
        with self.lock:
            if self.new_frame_available:
                self.new_frame_available = False
                return self.latest_frame
            return None

    def set_text_alert(self, duration=4.0):
        with self.lock:
            self.text_alert_expires = time.time() + duration

    def get_snapshot(self):
        with self.lock:
            # 1. Expiration Check
            if time.time() > self.text_alert_expires:
                if self.data["clinical_flag"] not in ["None", "N/A"]:
                    self.data["clinical_flag"] = "None"
            
            # 2. Multimodal Inference Logic (The "Fusion" Brain)
            face = str(self.data["face_emotion"]).lower()
            voice = str(self.data["vocal_emotion"]).lower()
            gaze = self.data["gaze_status"]
            
            is_distressed = False
            
            # Logic A: Anxiety (Fear + Avoidance)
            if face == "fear" and gaze == "Looking Away":
                is_distressed = True
                # Ideally we'd update a specific flag, but for now we trigger the main alert
            
            # Logic B: Anger/Aggression (Angry Face + Angry Voice)
            if face == "angry" or voice == "angry":
                is_distressed = True

            # Logic C: Text Trigger
            is_text_risk = time.time() < self.text_alert_expires
            
            self.data["alert_active"] = (is_distressed or is_text_risk)
            return self.data.copy()

state_manager = SharedState()

# ---------------- FASTAPI APP ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HELPER FUNCTIONS ----------------

def landmarks_to_np(landmarks, img_w, img_h):
    return np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks], dtype=np.float32)

def get_ear(landmarks, eye_indices, img_w, img_h):
    try:
        pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices])
        v_dist = np.linalg.norm(pts[2] - pts[3])
        h_dist = np.linalg.norm(pts[0] - pts[1])
        return (v_dist / h_dist) if h_dist > 1e-6 else 0.0
    except: return 0.0

def eye_iris_center(landmarks, eye_indices, iris_indices, img_w, img_h):
    try:
        pts_iris = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in iris_indices])
        iris_center = pts_iris.mean(axis=0)
        pts_eye = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices])
        left, right = pts_eye[0], pts_eye[1]
        top, bottom = pts_eye[2], pts_eye[3]
        eye_width = np.linalg.norm(right - left)
        eye_height = np.linalg.norm(top - bottom)
        if eye_width < 1e-6 or eye_height < 1e-6: return 0.0, 0.0
        dx = np.dot(iris_center - (left + right)/2.0, right - left) / (eye_width**2)
        dy = (iris_center[1] - (top[1] + bottom[1])/2.0) / eye_height
        return float(np.clip(dx * 2.0, -1.0, 1.0)), float(np.clip(dy * 2.0, -1.0, 1.0)), iris_center
    except: return 0.0, 0.0, None

def estimate_head_pose(landmarks, img_w, img_h):
    try:
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        idx = [1, 152, 33, 263, 61, 291]
        image_points = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in idx], dtype=np.float64)
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0.0, 0.0, 0.0
        rot_mat, _ = cv2.Rodrigues(rvec)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0]) * 180.0 / np.pi
        pitch = np.arcsin(-rot_mat[2, 0]) * 180.0 / np.pi
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2]) * 180.0 / np.pi
        return float(yaw), float(pitch), float(roll)
    except: return 0.0, 0.0, 0.0

class TemporalSmoother:
    def __init__(self, window=6, ema_alpha=0.35, min_dwell=3, conf_margin=0.05, spike_threshold=0.60):
        self.window = window
        self.labels = deque(maxlen=window)
        self.probs = deque(maxlen=window)
        self.ema_alpha = ema_alpha
        self.ema_prob = None
        self.current_label = None
        self.min_dwell = min_dwell
        self.conf_margin = conf_margin
        self.spike_threshold = spike_threshold

    def update(self, label, prob):
        if self.current_label is not None and label != self.current_label and prob >= self.spike_threshold:
            self.current_label = label
            self.labels.clear(); self.probs.clear()
        self.labels.append(label)
        self.probs.append(float(prob))
        if self.ema_prob is None: self.ema_prob = float(prob)
        else: self.ema_prob = (1.0 - self.ema_alpha) * self.ema_prob + self.ema_alpha * float(prob)
        majority_label = Counter(self.labels).most_common(1)[0][0] if self.labels else None
        stable_enough = len(self.labels) >= self.min_dwell and all(l == majority_label for l in list(self.labels)[-self.min_dwell:])
        if self.current_label is None: self.current_label = majority_label
        else:
            if (majority_label != self.current_label and stable_enough):
                self.current_label = majority_label
        return self.current_label, self.ema_prob
    
    def reset(self):
        self.labels.clear()
        self.probs.clear()
        self.ema_prob = None
        self.current_label = None

# ---------------- BACKGROUND ANALYSIS THREADS ----------------

clinical_schema = {
    "type": "OBJECT",
    "properties": {
        "sentiment": {"type": "STRING", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL", "SARCASTIC"]},
        "clinical_flag": {"type": "STRING"},
    },
    "required": ["sentiment", "clinical_flag"]
}

def analyze_text_clinically(text):
    if not gemini_model or not text: return "N/A", "N/A"
    
    # --- CLINICAL PROMPT (Updated with your ODD/MDD/Anxiety definitions) ---
    prompt = f"""
    Analyze this patient statement: "{text}".
    
    Detect clinical markers based on these definitions:
    1. **Anxiety/Distress**: Expressions of worry ("I'm worried"), feeling overwhelmed, rapid onset ("It's not the same"), or loss ("I miss...").
    2. **ODD/Anger**: Externalizing blame ("It's their fault"), annoyance ("They annoy me"), defiance, arguing with authority.
    3. **Depression/Apathy**: Hopelessness ("Better off without me"), isolation, flat/short responses ("Whatever", "Fine").
    
    Output strict JSON:
    - sentiment: POSITIVE, NEGATIVE, NEUTRAL, or SARCASTIC.
    - clinical_flag: The specific category identified (e.g., "Anxiety Indicator", "Distress", "ODD/Anger", "Depression/Apathy") or "None".
    """
    
    try:
        gen_config = genai.GenerationConfig(response_mime_type="application/json", response_schema=clinical_schema)
        response = gemini_model.generate_content(prompt, generation_config=gen_config)
        data = json.loads(response.text)
        return data.get("sentiment", "N/A"), data.get("clinical_flag", "None")
    except: return "Neutral", "None"

def map_vocal_label(label):
    mapping = {"neu":"Neutral","hap":"Happy","ang":"Angry","sad":"Sad","exc":"Excited","fea":"Fear","sur":"Surprise"}
    return mapping.get(label, label)

def check_safety_keywords(text):
    # Expanded list based on your docs (ODD/Depression markers)
    keywords = [
        "angry", "hate", "kill", "die", "suicide", "hurt", "pain", 
        "stupid", "idiot", "shut up", "hit", "punch", "blood",
        "fault", "annoy", "fair", "whatever", "fine", "worried", "scared"
    ]
    return any(w in text.lower() for w in keywords)

# --- AUDIO THREAD ---
def audio_processing_thread():
    print("[THREAD] Audio Listener Started")
    try:
        if not os.path.exists("model"): return
        model = Model("model"); rec = KaldiRecognizer(model, 16000)
    except: return
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    stream.start_stream()
    audio_buffer = bytearray()
    
    while True:
        try:
            data = stream.read(2000, exception_on_overflow=False)
            audio_buffer.extend(data)
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get('text', '')
                if text:
                    print(f"[FINAL] {text}")
                    
                    # 1. Immediate Keyword Check
                    if check_safety_keywords(text):
                        state_manager.update("clinical_flag", "Potential Risk Marker")
                        state_manager.set_text_alert(4.0)
                    
                    state_manager.update("text", text)
                    
                    # 2. Gemini Analysis (Async)
                    sent, flag = analyze_text_clinically(text)
                    
                    # 3. Vocal Analysis
                    vocal_emo = "Silence"
                    if vocal_pipeline and len(audio_buffer) > 0:
                        audio_float = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                        res = vocal_pipeline(audio_float, sampling_rate=16000)
                        if res: vocal_emo = map_vocal_label(res[0]['label'])
                    
                    state_manager.update("sentiment", sent)
                    state_manager.update("clinical_flag", flag)
                    state_manager.update("vocal_emotion", vocal_emo)
                    
                    if flag not in ["None", "N/A"] or sent == "NEGATIVE":
                        state_manager.set_text_alert(4.0)
                    
                    audio_buffer = bytearray()
        except: time.sleep(0.1)

# --- EMOTION THREAD ---
def emotion_analysis_thread():
    print("[THREAD] Background Emotion Analyzer Started")
    emotion_smoother = TemporalSmoother()
    
    while True:
        frame = state_manager.get_frame_for_analysis()
        
        if frame is not None:
            try:
                if fer:
                    # HSEmotion
                    emo, scores = fer.predict_emotions(frame, logits=False)
                    face_emo = emo if emo else "N/A"
                    face_conf = np.max(scores) if scores is not None else 0.0
                else:
                    # DeepFace
                    small_frame = cv2.resize(frame, (224, 224))
                    objs = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                    face_emo = objs[0]['dominant_emotion']
                    face_conf = objs[0]['emotion'][face_emo] / 100.0

                final_emo, final_conf = emotion_smoother.update(face_emo, face_conf)
                state_manager.update("face_emotion", final_emo)
                state_manager.update("face_conf", round(final_conf, 2))

            except Exception: pass
        else:
            # NO FACE DETECTED -> FORCE RESET
            emotion_smoother.reset()
            state_manager.update("face_emotion", "No Face")
            state_manager.update("face_conf", 0.0)
            time.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=audio_processing_thread, daemon=True).start()
    threading.Thread(target=emotion_analysis_thread, daemon=True).start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = state_manager.get_snapshot()
            await websocket.send_json(data)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect: pass

# --- VIDEO LOGIC ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    gaze_history = deque(maxlen=GAZE_SMOOTHING)
    calib_offset = (0.0, 0.0)
    calib_buffer = []
    calibrated = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        status_text = "No Face"
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Crop Logic
            pts = landmarks_to_np(lm, w, h)
            x_min, y_min = np.min(pts, axis=0).astype(int)
            x_max, y_max = np.max(pts, axis=0).astype(int)
            x_min, y_min = max(0, x_min-20), max(0, y_min-20)
            x_max, y_max = min(w, x_max+20), min(h, y_max+20)
            
            if x_max > x_min and y_max > y_min:
                face_crop = frame[y_min:y_max, x_min:x_max].copy()
                state_manager.set_frame_for_analysis(face_crop)
            else:
                 state_manager.set_frame_for_analysis(None)

            # Gaze Logic
            eye_idxs = [(33, 133, 159, 145), (263, 362, 386, 374)]
            iris_idxs = [list(range(468, 473)), list(range(473, 478))]
            
            lx, ly, l_iris_center = eye_iris_center(lm, eye_idxs[0], iris_idxs[0], w, h)
            rx, ry, r_iris_center = eye_iris_center(lm, eye_idxs[1], iris_idxs[1], w, h)
            gx, gy = (lx+rx)/2, (ly+ry)/2
            
            gaze_history.append((gx, gy))
            avg_g = np.mean(gaze_history, axis=0)
            
            if not calibrated:
                calib_buffer.append(avg_g)
                if len(calib_buffer) > 30:
                    arr = np.array(calib_buffer)
                    calib_offset = (arr[:,0].mean(), arr[:,1].mean())
                    calibrated = True
            
            final_gx = avg_g[0] - calib_offset[0]
            final_gy = avg_g[1] - calib_offset[1]
            
            ear = (get_ear(lm, eye_idxs[0], w, h) + get_ear(lm, eye_idxs[1], w, h)) / 2
            yaw, pitch, _ = estimate_head_pose(lm, w, h)
            
            if ear < EAR_THRESHOLD:
                status_text = "Eyes Closed"
            elif abs(yaw) > HEAD_YAW_THRESHOLD:
                status_text = "Looking Away" 
            elif abs(final_gx) < LOOK_THRESHOLD_X and abs(final_gy) < LOOK_THRESHOLD_Y:
                status_text = "Looking at Screen"
            else:
                status_text = "Looking Away"
            
            state_manager.update("gaze_status", status_text)
            state_manager.update("blink_state", "Closed" if ear < EAR_THRESHOLD else "Open")

            if l_iris_center is not None:
                cv2.circle(frame, (int(l_iris_center[0]), int(l_iris_center[1])), 2, (0, 0, 255), -1)
            if r_iris_center is not None:
                cv2.circle(frame, (int(r_iris_center[0]), int(r_iris_center[1])), 2, (0, 0, 255), -1)
        else:
             state_manager.set_frame_for_analysis(None)
             state_manager.update("gaze_status", "No Face")

        snap = state_manager.get_snapshot()
        cv2.putText(frame, f"Gaze: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)