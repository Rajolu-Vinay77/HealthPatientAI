import cv2
import time
import numpy as np
import mediapipe as mp
import threading
import queue
import json
import os
import asyncio
import csv
import io
import math
import re
import random
from collections import deque, Counter
from datetime import datetime, timedelta
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIGURATION ----------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SESSION_API_KEY = os.getenv("SESSION_API_KEY", "dev-secret") 

# Fail-fast check for production credentials
if os.getenv("ENV") == "production" and SESSION_API_KEY == "dev-secret":
    raise RuntimeError("CRITICAL: SESSION_API_KEY must be changed in production environment.")

# Global Event for Graceful Shutdown
STOP_EVENT = threading.Event()

# Ensure storage directory exists
RECORD_DIR = "session_records"
os.makedirs(RECORD_DIR, exist_ok=True)

# Safe Import for Vosk (Audio)
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    print("[WARN] 'vosk' library not found. Audio transcription will be disabled.")
    VOSK_AVAILABLE = False
    Model, KaldiRecognizer = None, None

# Safe Import for Audio/ML libs
import pyaudio
from deepface import DeepFace
import google.generativeai as genai

# Optional Transformers
try:
    from transformers import pipeline
    vocal_pipeline = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
    print("[INFO] Vocal Emotion Model Loaded")
except:
    vocal_pipeline = None
    print("[WARN] Transformers library missing. Vocal analysis disabled.")

# Configure Gemini
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_name = 'models/gemini-2.5-flash'
        gemini_model = genai.GenerativeModel(model_name)
        print(f"[INFO] Gemini API initialized: {model_name}")
    except Exception as e:
        print(f"[ERROR] Gemini Config Failed: {e}")
else:
    print("[WARN] GEMINI_API_KEY not set. Falling back to local heuristics.")

# ---------------- CONSTANTS ----------------
LOOK_THRESHOLD_X = 0.22
LOOK_THRESHOLD_Y = 0.20
HEAD_YAW_THRESHOLD = 25.0
EAR_THRESHOLD = 0.12
GAZE_SMOOTHING = 6
MAX_HISTORY_ROWS = 3600 # 1 hour buffer
EMOTION_POLL_INTERVAL = 0.5 # Run DeepFace every 0.5s (reduces CPU load)
MAX_DEEPFACE_ERRORS = 5 # Circuit breaker threshold

# ---------------- SHARED STATE & RECORDER ----------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "status": "System Ready",
            "face_emotion": "Neutral",
            "face_conf": 0.0,
            "gaze_status": "N/A",
            "blink_state": "Open",
            "blink_rate": 0.0,        # BPM (Smoothed for UI)
            "blink_total": 0,         # NEW: Raw Count (Lossless for CSV)
            "vocal_emotion": "N/A",
            "text": "",
            "sentiment": "Neutral",
            "clinical_flag": "None",
            "alert_active": False,
            "response_latency": 0.0
        }
        self.text_alert_expires = 0.0
        self.latest_frame = None 
        self.new_frame_available = False

        self.patient_context = {
            "age": 30,          # Default
            "gender": "Unknown",
            "culture": "General"
        }
        
        # --- Session Recording Data ---
        self.is_recording = False
        self.session_start_time = time.time()
        self.history_log = deque(maxlen=MAX_HISTORY_ROWS) 
        self.full_transcript = [] 
        self.consent_metadata = {}

        # --- BIOMETRIC TRACKERS ---
        self.blink_timestamps = deque(maxlen=100) 
        self.last_speech_end_time = time.time()   

    def update(self, key, value):
        with self.lock:
            self.data[key] = value
            # Transcript accumulation
            if key == "text" and value:
                ts = round(time.time() - self.session_start_time, 2)
                # Simple dedup
                if not self.full_transcript or self.full_transcript[-1]['text'] != value:
                    self.full_transcript.append({
                        "time": ts, 
                        "speaker": "Patient", 
                        "text": value
                    })
    
    # --- BIOMETRIC LOGIC ---
    def reset_biometrics(self):
        """Clears rate calculation but KEEPS total count."""
        with self.lock:
            self.blink_timestamps.clear()
            self.data["blink_rate"] = 0.0

    def register_blink(self):
        """Called ONLY when a specific blink event occurs."""
        with self.lock:
            self.blink_timestamps.append(time.time())
            self.data["blink_total"] += 1 # Increment raw count

    def update_bpm_logic(self):
        """Called EVERY FRAME to ensure the rate decays over time."""
        with self.lock:
            now = time.time()
            
            # 1. Prune blinks older than 60 seconds
            cutoff = now - 60
            while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
                self.blink_timestamps.popleft()
            
            # 2. Calculate Duration
            if self.is_recording:
                duration = now - self.session_start_time
            else:
                duration = 60.0 if len(self.blink_timestamps) > 0 else 1.0

            count = len(self.blink_timestamps)

            # 3. UI STABILIZATION
            if duration < 5:
                # HIDE NOISE: If session just started (<5s), show 0.0
                self.data["blink_rate"] = 0.0
            elif duration < 60:
                # Extrapolate, but cap at 60 to prevent visual glitches
                raw_bpm = count * (60 / duration)
                self.data["blink_rate"] = round(min(raw_bpm, 60.0), 1)
            else:
                # Standard Window
                self.data["blink_rate"] = float(count)
            
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
            if time.time() > self.text_alert_expires:
                if self.data["clinical_flag"] not in ["None", "N/A"]:
                    self.data["clinical_flag"] = "None"
            
            is_angry_face = str(self.data["face_emotion"]).lower() == "angry"
            is_angry_voice = str(self.data["vocal_emotion"]).lower() == "angry"
            is_risky_text = time.time() < self.text_alert_expires
            
            self.data["alert_active"] = (is_angry_face or is_angry_voice or is_risky_text)
            return self.data.copy()
    
    # --- History Logger ---
    def record_snapshot(self):
        with self.lock:
            if not self.is_recording: return
            
            elapsed = round(time.time() - self.session_start_time, 2)
            snapshot = self.data.copy()
            
            # Check if face is present
            face_detected = snapshot["face_emotion"] != "No Face"

            face_emo = str(snapshot["face_emotion"]).lower()
            behavior = "Neutral"
            if snapshot["alert_active"]: behavior = "Stressed/Escalated"
            elif face_emo == "happy": behavior = "Positive"
            elif snapshot["gaze_status"] == "Looking Away": behavior = "Disengaged/Avoidant"
            
            # DATA INTEGRITY: Record None if face lost
            blink_val = snapshot["blink_rate"] if face_detected else None

            log_entry = {
                "Timestamp": elapsed,
                "Speaker": "Patient", 
                "Text": snapshot["text"] if snapshot["text"] else "—",
                "Emotion": snapshot["face_emotion"],
                "Tone": snapshot["vocal_emotion"],
                "Behavior": behavior,
                "Confidence": snapshot["face_conf"],
                "BlinkRate": blink_val,
                "BlinkCount": snapshot["blink_total"], 
                "Latency": snapshot["response_latency"],
                "Notes": snapshot["clinical_flag"]
            }
            
            if self.data["text"]:
                self.data["text"] = "" 
                
            self.history_log.append(log_entry)

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

# ---------------- SECURITY ----------------
async def require_api_key(x_api_key: str = Header(None)):
    """Protects sensitive endpoints."""
    if SESSION_API_KEY and x_api_key != SESSION_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# ---------------- HELPER FUNCTIONS ----------------

def redact_pii(text):
    """Simple regex to strip emails and phone numbers before sending to AI."""
    PII_PATTERNS = [
        r'\b[\w\.-]+@[\w\.-]+\.\w+\b',          # Email
        r'\b\d{10}\b',                          # 10-digit phone
        r'\b(?:\d{3}[-.\s]?){2}\d{4}\b',        # Phone with separators
    ]
    for p in PII_PATTERNS:
        text = re.sub(p, "[REDACTED]", text, flags=re.IGNORECASE)
    return text

def check_safety_keywords(text):
    keywords = ["angry", "hate", "kill", "die", "suicide", "hurt", "pain", "stupid", "idiot", "fault", "annoy"]
    return any(w in text.lower() for w in keywords)

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
        if eye_width < 1e-6 or eye_height < 1e-6: return 0.0, 0.0, None
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
    def __init__(self, window=3, ema_alpha=0.5, min_dwell=1, spike_threshold=0.60):
        self.window = window
        self.labels = deque(maxlen=window)
        self.probs = deque(maxlen=window)
        self.ema_alpha = ema_alpha
        self.ema_prob = None
        self.current_label = None
        self.min_dwell = min_dwell
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

# ---------------- ANALYSIS LOGIC ----------------

CLINICAL_EXAMPLES = []
def load_clinical_examples(filepath="telepsych_finetuning.jsonl"):
    examples = []
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                user_text = data['messages'][0]['content'].replace('Analyze this patient statement: ', '').strip('"')
                model_json = data['messages'][1]['content']
                examples.append(f"Patient: \"{user_text}\"\nAnalysis: {model_json}")
        return examples
    except Exception: return []

CLINICAL_EXAMPLES = load_clinical_examples()

def analyze_text_clinically(text):
    if not gemini_model or not text: return "N/A", "N/A"
    safe_text = redact_pii(text)
    
    # Inject Biometrics
    snapshot = state_manager.get_snapshot()
    biometrics = f"Blink Rate: {snapshot['blink_rate']} bpm, Latency: {snapshot['response_latency']}s"

    # Few-Shot Logic
    examples_context = ""
    if CLINICAL_EXAMPLES:
        subset = random.sample(CLINICAL_EXAMPLES, k=min(5, len(CLINICAL_EXAMPLES)))
        examples_context = "\n".join(subset)

    prompt = f"""
    You are an expert clinical psychiatrist AI.
    
    ### BIOMETRIC DATA
    {biometrics}
    (Note: Latency > 2s indicates hesitation/depression. Blink Rate < 10 is low, > 30 is high.)

    ### PATIENT STATEMENT
    "{safe_text}"
    
    ### CLINICAL EXAMPLES (Style Guide)
    {examples_context}
    
    ### TASK
    Analyze for ODD, Depression, or Anxiety markers. Use biometrics to support your finding.
    Output JSON: {{"sentiment": "...", "clinical_flag": "..."}}
    """
    try:
        gen_config = genai.GenerationConfig(response_mime_type="application/json")
        response = gemini_model.generate_content(prompt, generation_config=gen_config)
        data = json.loads(response.text)
        return data.get("sentiment", "N/A"), data.get("clinical_flag", "None")
    except: return "Neutral", "None"

def map_vocal_label(label):
    mapping = {"neu":"Neutral","hap":"Happy","ang":"Angry","sad":"Sad","exc":"Excited","fea":"Fear","sur":"Surprise"}
    return mapping.get(label, label)

def generate_local_annotations(transcript_text):
    lines = [ln.strip() for ln in transcript_text.splitlines() if ln.strip()]
    if not lines: lines = ["(No speech detected)"]
    annotated_lines = []
    for ln in lines:
        note = "neutral"
        if check_safety_keywords(ln): note = "Potential Risk Marker"
        annotated_lines.append(f"{ln} (Noted: {note})")
    report = f"### Overview (Local Fallback)\nSession processed using local heuristics."
    return "\n".join(annotated_lines), report

def call_gemini_with_retry(prompt, attempts=3):
    if not gemini_model: return None
    final_report_schema = {
        "type": "OBJECT",
        "properties": {
            "transcript_tagged": {"type": "STRING"},
            "report_content": {"type": "STRING"}
        },
        "required": ["transcript_tagged", "report_content"]
    }
    gen_config = genai.GenerationConfig(response_mime_type="application/json", response_schema=final_report_schema, max_output_tokens=2000)
    for i in range(attempts):
        try:
            return gemini_model.generate_content(prompt, generation_config=gen_config)
        except Exception: time.sleep(2 ** i)
    raise RuntimeError("Gemini failed after max retries")

# ---------------- THREADS ----------------

def audio_processing_thread():
    print("[THREAD] Audio Listener Started")
    if not VOSK_AVAILABLE or not os.path.exists("model"): return
    try:
        model = Model("model"); rec = KaldiRecognizer(model, 16000)
    except: return

    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()
        audio_buffer = bytearray()
        
        while not STOP_EVENT.is_set():
            try:
                data = stream.read(2000, exception_on_overflow=False)
                audio_buffer.extend(data)
                
                if rec.AcceptWaveform(data):
                    now = time.time()
                    latency = round(now - state_manager.last_speech_end_time, 2)
                    state_manager.update("response_latency", latency)
                    
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        print(f"[FINAL] {text} (Latency: {latency}s)")
                        state_manager.last_speech_end_time = time.time()

                        if check_safety_keywords(text):
                            state_manager.update("clinical_flag", "Keyword Alert")
                            state_manager.set_text_alert(4.0)
                        state_manager.update("text", text)
                        sent, flag = analyze_text_clinically(text)
                        vocal_emo = "Silence"
                        if vocal_pipeline and len(audio_buffer) > 0:
                            audio_float = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                            res = vocal_pipeline(audio_float, sampling_rate=16000)
                            if res: vocal_emo = map_vocal_label(res[0]['label'])
                        
                        state_manager.update("sentiment", sent)
                        state_manager.update("vocal_emotion", vocal_emo)
                        if flag not in ["None", "N/A"]:
                            state_manager.update("clinical_flag", flag)
                            state_manager.set_text_alert(4.0)
                        audio_buffer = bytearray()
            except Exception: time.sleep(0.1)
    finally:
        if stream:
            try: stream.stop_stream(); stream.close()
            except: pass
        try: p.terminate()
        except: pass

def emotion_analysis_thread():
    print("[THREAD] Background Emotion Analyzer Started")
    emotion_smoother = TemporalSmoother(window=3, min_dwell=1)
    last_run = 0.0
    error_count = 0
    
    while not STOP_EVENT.is_set():
        if error_count >= MAX_DEEPFACE_ERRORS: time.sleep(5); continue
        now = time.time()
        if now - last_run < EMOTION_POLL_INTERVAL: time.sleep(0.1); continue

        frame = state_manager.get_frame_for_analysis()
        if frame is not None:
            last_run = now
            try:
                small_frame = cv2.resize(frame, (224, 224))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                objs = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
                if objs:
                    res = objs[0] if isinstance(objs, list) else objs
                    raw_emo = res['dominant_emotion']
                    raw_conf = res['emotion'][raw_emo] / 100.0
                    final_emo, final_conf = emotion_smoother.update(raw_emo, raw_conf)
                    state_manager.update("face_emotion", final_emo)
                    state_manager.update("face_conf", round(final_conf, 2))
                    error_count = 0 
            except Exception as e:
                error_count += 1
        else:
            time.sleep(0.1)

def recording_thread():
    print("[THREAD] Session Recorder Started")
    while not STOP_EVENT.is_set():
        start_ts = time.time()
        state_manager.record_snapshot()
        time.sleep(max(0.0, 1.0 - (time.time() - start_ts)))

@app.on_event("startup")
async def startup_event():
    print(f"[INFO] Startup. VOSK={VOSK_AVAILABLE}, GEMINI={'Enabled' if gemini_model else 'Disabled'}")
    try:
        files = sorted([os.path.join(RECORD_DIR, f) for f in os.listdir(RECORD_DIR)], key=os.path.getmtime)
        if len(files) > 50:
            for f in files[:-50]: os.remove(f)
    except Exception: pass
    threading.Thread(target=audio_processing_thread, daemon=True).start()
    threading.Thread(target=emotion_analysis_thread, daemon=True).start()
    threading.Thread(target=recording_thread, daemon=True).start()

@app.on_event("shutdown")
def shutdown_event():
    print("[INFO] Shutting down...")
    STOP_EVENT.set()

# ---------------- ENDPOINTS ----------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = state_manager.get_snapshot()
            await websocket.send_json(data)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect: pass

def safe_frame_generator(gen):
    try:
        for frame in gen: yield frame
    except Exception as e: print(f"[ERROR] Video Feed Error: {e}"); return

def load_patient_baseline():
    """
    Scans the 'session_records' directory and calculates the average
    BlinkRate and Latency from all previous session CSVs.
    """
    blink_rates = []
    latencies = []
    
    if not os.path.exists(RECORD_DIR):
        return {"avg_blink": None, "avg_latency": None}

    # Iterate over all CSV files
    for filename in os.listdir(RECORD_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(RECORD_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Collect all valid rows from this file
                    file_blinks = []
                    file_lats = []
                    for row in reader:
                        # Parse BlinkRate
                        try:
                            b_rate = row.get("BlinkRate")
                            if b_rate and b_rate != "None" and float(b_rate) > 0:
                                file_blinks.append(float(b_rate))
                        except: pass
                        
                        # Parse Latency
                        try:
                            lat = row.get("Latency")
                            if lat and lat != "None" and float(lat) > 0:
                                file_lats.append(float(lat))
                        except: pass
                    
                    # Add this session's average to the history
                    if file_blinks: 
                        blink_rates.append(sum(file_blinks) / len(file_blinks))
                    if file_lats:
                        latencies.append(sum(file_lats) / len(file_lats))
                        
            except Exception as e:
                print(f"[WARN] Could not read baseline from {filename}: {e}")

    # Calculate Global Averages
    baseline_blink = round(sum(blink_rates) / len(blink_rates), 1) if blink_rates else 18.0 # Default healthy
    baseline_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.5   # Default healthy
    
    return {
        "avg_blink": baseline_blink,
        "avg_latency": baseline_latency,
        "sessions_count": len(blink_rates)
    }

@app.get("/video_feed")
def video_feed():
    gen = generate_frames()
    if gen is None: raise HTTPException(status_code=503, detail="Camera not available")
    return StreamingResponse(safe_frame_generator(gen), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/healthz")
def health_check():
    return {"status": "ok", "recording": state_manager.is_recording, "gemini": bool(gemini_model)}

# Update Pydantic model or accept query params. Here using query params for simplicity.
@app.post("/start_session")
async def start_session(
    age: int = 30, 
    gender: str = "Unknown", 
    culture: str = "General", 
    api_key: str = Depends(require_api_key)
):
    with state_manager.lock:
        # Store Context
        state_manager.patient_context = {"age": age, "gender": gender, "culture": culture}
        
        # Reset State
        state_manager.is_recording = True
        state_manager.session_start_time = time.time()
        state_manager.history_log.clear()
        state_manager.full_transcript.clear()
        state_manager.blink_timestamps.clear()
        state_manager.data["blink_total"] = 0
        state_manager.last_speech_end_time = time.time()
        
    return JSONResponse({
        "status": "Session Started", 
        "context": state_manager.patient_context
    })

@app.post("/stop_session")
async def stop_session(api_key: str = Depends(require_api_key)):
    state_manager.is_recording = False
    
    with state_manager.lock:
        rows = list(state_manager.history_log)
        transcript_data = list(state_manager.full_transcript)

    # 1. Calculate "Ground Truth" Statistics in Python (Not AI)
    # Filter out None/0 values to get the REAL biometric average
    valid_blinks = [r["BlinkRate"] for r in rows if r["BlinkRate"] is not None and r["BlinkRate"] > 0]
    avg_blink = round(sum(valid_blinks) / len(valid_blinks), 1) if valid_blinks else "N/A"
    
    valid_latency = [r["Latency"] for r in rows if r["Latency"] is not None and r["Latency"] > 0]
    avg_latency = round(sum(valid_latency) / len(valid_latency), 1) if valid_latency else "N/A"

    # Smart Emotion Calculation: Find the most frequent emotion that ISN'T "Neutral" or "No Face"
    emotions = [r["Emotion"] for r in rows if r["Emotion"] not in ["No Face", "neutral", "Neutral"]]
    if not emotions: 
        emotions = [r["Emotion"] for r in rows if r["Emotion"] != "No Face"] # Fallback to include neutral
    
    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"

    # 2. Generate CSV
    csv_output = io.StringIO()
    fieldnames = ["Timestamp", "Speaker", "Text", "Emotion", "Tone", "Behavior", "Confidence", "BlinkRate", "BlinkCount", "Latency", "Notes"]
    dict_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    dict_writer.writeheader()
    dict_writer.writerows(rows)
    csv_string = csv_output.getvalue()

    # Save to Disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RECORD_DIR}/session_{timestamp}.csv"
    try:
        with open(filename, "w", newline='', encoding='utf-8') as f: f.write(csv_string)
        print(f"[INFO] Saved session record to {filename}")
    except Exception: pass

    # Prepare Transcript
    transcript_lines = [f"[{e['time']}s] Patient: {e['text']}" for e in transcript_data]
    full_text_log = "\n".join(transcript_lines) or "(No speech detected)"
    safe_transcript = redact_pii(full_text_log)

    # 3. THE "STRICT" PROMPT
    # We clearly separate "GROUND TRUTH" from "ANALYSIS"
    prompt = f"""
    You are an expert clinical psychiatrist AI.
    
    ### GROUND TRUTH BIOMETRICS (Do Not Recalculate)
    * These values have been calculated by sensors. You must use them as facts.
    - Dominant Emotion: {dominant_emotion}
    - Average Blink Rate: {avg_blink} bpm
    - Average Response Latency: {avg_latency}s
    
    ### PATIENT TRANSCRIPT
    {safe_transcript}
    
    ### TASK
    Write a clinical session summary.
    1. **Emotional Analysis:** State the Dominant Emotion exactly as listed above ({dominant_emotion}). Analyze what this suggests.
    2. **Physiological Indicators:** Discuss the Blink Rate ({avg_blink}). Is it low (<10) or high (>30)?
    3. **Interaction Dynamics:** Discuss the Latency ({avg_latency}). Is it delayed (>2s)?
    4. **Clinical Impression:** Based *only* on the data above, does this match Anhedonia (Neutral/Sad), Agitation (Angry/Fear), or Anxiety (High Blink)?

    ### OUTPUT FORMAT (Strict JSON)
    {{
      "transcript_tagged": "Verbatim transcript with (Noted: ...) tags",
      "report_content": "Professional clinical summary text..."
    }}
    """

    report_data = {"transcript_tagged": "", "report_content": "", "csv": csv_string}
    
    try:
        response = call_gemini_with_retry(prompt)
        if response:
            try:
                json_res = json.loads(response.text)
                report_data["transcript_tagged"] = json_res.get("transcript_tagged", "")
                report_data["report_content"] = json_res.get("report_content", "")
            except Exception:
                report_data["report_content"] = response.text
                report_data["transcript_tagged"] = "Format Error"
    except Exception as e:
        report_data["report_content"] = f"AI Generation Failed: {e}"
    
    # Fallback
    if not report_data["report_content"]:
        annotated, local_report = generate_local_annotations(safe_transcript)
        report_data["transcript_tagged"] = annotated
        report_data["report_content"] = local_report

    return JSONResponse(content=report_data)

@app.get("/download_csv")
def download_csv(api_key: str = Depends(require_api_key)):
    with state_manager.lock:
        rows = list(state_manager.history_log)
    csv_output = io.StringIO()
    fieldnames = ["Timestamp", "Speaker", "Text", "Emotion", "Tone", "Behavior", "Confidence", "BlinkRate", "BlinkCount", "Latency", "Notes"]
    dict_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    dict_writer.writeheader()
    dict_writer.writerows(rows)
    return Response(content=csv_output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=session_log.csv"})

@app.post("/clear_session")
async def clear_session(api_key: str = Depends(require_api_key)):
    with state_manager.lock:
        state_manager.history_log.clear()
        state_manager.full_transcript.clear()
        state_manager.is_recording = False
    return JSONResponse({"status": "Session Cleared", "recording": False})

# --- VIDEO LOGIC (ROBUST + DECAY) ---
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): return None
    for _ in range(5): cap.read()
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        gaze_history = deque(maxlen=GAZE_SMOOTHING)
        ear_history = deque(maxlen=3)
        calib_offset = (0.0, 0.0)
        calib_buffer = []
        calibrated = False
        read_failures = 0
        eyes_previously_open = True 

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    read_failures += 1
                    if read_failures > 10: break 
                    continue
                read_failures = 0 
                
                # --- NEW: ALWAYS UPDATE BPM DECAY ---
                state_manager.update_bpm_logic()

                try:
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    status_text = "No Face"
                    
                    if results.multi_face_landmarks:
                        state_manager.set_frame_for_analysis(frame)
                        lm = results.multi_face_landmarks[0].landmark
                        eye_idxs = [(33, 133, 159, 145), (263, 362, 386, 374)]
                        iris_idxs = [list(range(468, 473)), list(range(473, 478))]
                        
                        # 1. Blink Detection
                        left_ear = get_ear(lm, eye_idxs[0], w, h)
                        right_ear = get_ear(lm, eye_idxs[1], w, h)
                        current_ear = (left_ear + right_ear) / 2.0
                        ear_history.append(current_ear)
                        avg_ear = sum(ear_history) / len(ear_history)
                        
                        SMOOTH_EAR_THRESHOLD = 0.21
                        is_blink = avg_ear < SMOOTH_EAR_THRESHOLD

                        # 2. Register Blink (Rising Edge)
                        if is_blink and eyes_previously_open:
                            state_manager.register_blink() 
                        eyes_previously_open = not is_blink

                        if is_blink:
                            status_text = "Eyes Closed"
                            state_manager.update("blink_state", "Closed")
                            state_manager.update("gaze_status", status_text)
                        else:
                            state_manager.update("blink_state", "Open")
                            lx, ly, l_iris_center = eye_iris_center(lm, eye_idxs[0], iris_idxs[0], w, h)
                            rx, ry, r_iris_center = eye_iris_center(lm, eye_idxs[1], iris_idxs[1], w, h)
                            if l_iris_center is not None and r_iris_center is not None:
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
                                yaw, pitch, _ = estimate_head_pose(lm, w, h)
                                if abs(yaw) > HEAD_YAW_THRESHOLD: status_text = "Looking Away"
                                elif abs(final_gx) < LOOK_THRESHOLD_X and abs(final_gy) < LOOK_THRESHOLD_Y: status_text = "Looking at Screen"
                                else: status_text = "Looking Away"
                                state_manager.update("gaze_status", status_text)
                                cv2.circle(frame, (int(l_iris_center[0]), int(l_iris_center[1])), 2, (0, 0, 255), -1)
                                cv2.circle(frame, (int(r_iris_center[0]), int(r_iris_center[1])), 2, (0, 0, 255), -1)
                            else:
                                state_manager.update("gaze_status", "Obstructed")
                    else:
                        state_manager.update("gaze_status", "No Face")
                        state_manager.update("face_emotion", "No Face")
                        state_manager.update("face_conf", 0.0)
                        state_manager.update("blink_state", "N/A")
                        state_manager.reset_biometrics() 

                except Exception: pass
                
                snap = state_manager.get_snapshot()
                cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Gaze: {snap['gaze_status']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"BPM: {snap.get('blink_rate', 0)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)