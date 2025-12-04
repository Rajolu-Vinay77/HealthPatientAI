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
        model_name = 'gemini-2.5-flash'
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
            "vocal_emotion": "N/A",
            "text": "",
            "sentiment": "Neutral",
            "clinical_flag": "None",
            "alert_active": False
        }
        self.text_alert_expires = 0.0
        self.latest_frame = None 
        self.new_frame_available = False
        
        # --- Session Recording Data ---
        self.is_recording = False
        self.session_start_time = time.time()
        self.history_log = deque(maxlen=MAX_HISTORY_ROWS) 
        self.full_transcript = [] 
        self.consent_metadata = {}

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
            
            # Robust Case-Insensitive Check
            face_emo = str(snapshot["face_emotion"]).lower()
            
            behavior = "Neutral"
            if snapshot["alert_active"]: behavior = "Stressed/Escalated"
            elif face_emo == "happy": behavior = "Positive"
            elif snapshot["gaze_status"] == "Looking Away": behavior = "Disengaged/Avoidant"
            
            log_entry = {
                "Timestamp": elapsed,
                "Speaker": "Patient", 
                "Text": snapshot["text"] if snapshot["text"] else "—",
                "Emotion": snapshot["face_emotion"],
                "Tone": snapshot["vocal_emotion"],
                "Behavior": behavior,
                "Confidence": snapshot["face_conf"],
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

# ---------------- CLINICAL LOGIC (IN-CONTEXT) ----------------

# Load examples once at startup
CLINICAL_EXAMPLES = []
def load_clinical_examples(filepath="telepsych_finetuning.jsonl"):
    """Reads the JSONL file to get 'real' DAIC-WOZ examples."""
    examples = []
    if not os.path.exists(filepath):
        print(f"[WARN] Training data '{filepath}' not found. Using default logic.")
        return []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Extract text and label
                user_text = data['messages'][0]['content'].replace('Analyze this patient statement: ', '').strip('"')
                model_json = data['messages'][1]['content']
                examples.append(f"Patient: \"{user_text}\"\nAnalysis: {model_json}")
        print(f"[INFO] Loaded {len(examples)} clinical examples for In-Context Learning.")
        return examples
    except Exception as e:
        print(f"[ERROR] Failed to load training data: {e}")
        return []

CLINICAL_EXAMPLES = load_clinical_examples()

def analyze_text_clinically(text):
    if not gemini_model or not text: return "N/A", "N/A"
    
    safe_text = redact_pii(text)
    
    # Dynamic Few-Shot Selection
    examples_context = ""
    if CLINICAL_EXAMPLES:
        # Pick 8 random examples to keep the prompt fresh
        subset = random.sample(CLINICAL_EXAMPLES, k=min(8, len(CLINICAL_EXAMPLES)))
        examples_context = "\n\n".join(subset)
    
    prompt = f"""
    You are an expert clinical psychiatrist AI.
    
    ### TASK
    Analyze the patient statement below for clinical markers of ODD, Depression, or Anxiety.
    
    ### CLINICAL DEFINITIONS
    1. **ODD (Oppositional Defiant Disorder)**: Easily annoyed, externalizes blame ("it's their fault"), defiant.
    2. **Adjustment Disorder**: Stressor response, "it's not the same", "I miss my...".
    3. **Depression**: Hopelessness, "better off without me", isolation, sleep/energy issues.
    
    ### REFERENCE EXAMPLES (Learn from these patterns)
    {examples_context}
    
    ### CURRENT PATIENT STATEMENT
    "{safe_text}"
    
    ### OUTPUT FORMAT
    Strict JSON: {{"sentiment": "...", "clinical_flag": "..."}}
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
        elif "um" in ln or "uh" in ln: note = "Hesitation"
        annotated_lines.append(f"{ln} (Noted: {note})")
    report = f"### Overview (Local Fallback)\nSession processed using local heuristics.\nTranscript lines: {len(lines)}"
    return "\n".join(annotated_lines), report

def call_gemini_with_retry(prompt, attempts=3):
    if not gemini_model: return None
    
    # We define the schema for the Final Report here
    final_report_schema = {
        "type": "OBJECT",
        "properties": {
            "transcript_tagged": {"type": "STRING"},
            "report_content": {"type": "STRING"}
        },
        "required": ["transcript_tagged", "report_content"]
    }
    
    gen_config = genai.GenerationConfig(
        response_mime_type="application/json", 
        response_schema=final_report_schema,
        max_output_tokens=2000
    )
    
    for i in range(attempts):
        try:
            return gemini_model.generate_content(prompt, generation_config=gen_config)
        except Exception as e:
            wait = 2 ** i
            print(f"[WARN] Gemini attempt {i+1} failed: {e} - Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Gemini failed after max retries")

# ---------------- THREADS ----------------

def audio_processing_thread():
    print("[THREAD] Audio Listener Started")
    if not VOSK_AVAILABLE or not os.path.exists("model"): return

    try:
        model = Model("model"); rec = KaldiRecognizer(model, 16000)
    except Exception as e:
        print(f"[ERROR] Audio Init Failed: {e}")
        return

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
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        print(f"[FINAL] {text}")
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
        if error_count >= MAX_DEEPFACE_ERRORS:
            time.sleep(5) 
            continue

        now = time.time()
        if now - last_run < EMOTION_POLL_INTERVAL:
            time.sleep(0.1)
            continue

        frame = state_manager.get_frame_for_analysis()
        if frame is not None:
            last_run = now
            try:
                small_frame = cv2.resize(frame, (224, 224))
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                objs = DeepFace.analyze(
                    rgb_frame, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='skip', 
                    silent=True
                )
                
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
                if error_count >= MAX_DEEPFACE_ERRORS:
                    print(f"[WARN] DeepFace disabled due to repeated errors: {e}")
        else:
            time.sleep(0.1)

def recording_thread():
    print("[THREAD] Session Recorder Started")
    while not STOP_EVENT.is_set():
        start_ts = time.time()
        state_manager.record_snapshot()
        process_time = time.time() - start_ts
        sleep_time = max(0.0, 1.0 - process_time)
        time.sleep(sleep_time)

@app.on_event("startup")
async def startup_event():
    print(f"[INFO] System Startup. VOSK={VOSK_AVAILABLE}, GEMINI={'Enabled' if gemini_model else 'Disabled'}")
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
    except Exception as e:
        print(f"[ERROR] Video Feed Error: {e}")
        return

@app.get("/video_feed")
def video_feed():
    gen = generate_frames()
    if gen is None: raise HTTPException(status_code=503, detail="Camera not available")
    return StreamingResponse(safe_frame_generator(gen), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/healthz")
def health_check():
    return {"status": "ok", "recording": state_manager.is_recording, "gemini": bool(gemini_model)}

@app.get("/session_status")
def session_status():
    with state_manager.lock:
        elapsed = 0
        if state_manager.is_recording:
            elapsed = round(time.time() - state_manager.session_start_time, 2)
        return {
            "is_recording": state_manager.is_recording,
            "elapsed_seconds": elapsed,
            "records_count": len(state_manager.history_log)
        }

@app.post("/record_consent")
async def record_consent(user_id: str = "anonymous", agreed: bool = True):
    state_manager.consent_metadata = {
        "user_id": user_id,
        "consented": agreed,
        "timestamp": datetime.utcnow().isoformat()
    }
    return {"status": "Consent Recorded", "user": user_id}

@app.post("/start_session")
async def start_session(api_key: str = Depends(require_api_key)):
    if not state_manager.consent_metadata.get("consented"):
        print("[WARN] Starting session without explicit consent log.")
    with state_manager.lock:
        state_manager.is_recording = True
        state_manager.session_start_time = time.time()
        state_manager.history_log.clear()
        state_manager.full_transcript.clear()
    return JSONResponse({"status": "Session Started", "recording": True})

@app.post("/clear_session")
async def clear_session(api_key: str = Depends(require_api_key)):
    with state_manager.lock:
        state_manager.history_log.clear()
        state_manager.full_transcript.clear()
        state_manager.is_recording = False
    return JSONResponse({"status": "Session Cleared", "recording": False})

@app.get("/download_csv")
def download_csv(api_key: str = Depends(require_api_key)):
    with state_manager.lock:
        rows = list(state_manager.history_log)
    csv_output = io.StringIO()
    fieldnames = ["Timestamp","Speaker","Text","Emotion","Tone","Behavior","Confidence","Notes"]
    writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return Response(content=csv_output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=session_log.csv"})

@app.post("/stop_session")
async def stop_session(api_key: str = Depends(require_api_key)):
    state_manager.is_recording = False
    
    with state_manager.lock:
        rows = list(state_manager.history_log)
        transcript_data = list(state_manager.full_transcript)
        sensor_summary = state_manager.data.copy()

    csv_output = io.StringIO()
    fieldnames = ["Timestamp", "Speaker", "Text", "Emotion", "Tone", "Behavior", "Confidence", "Notes"]
    dict_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
    dict_writer.writeheader()
    dict_writer.writerows(rows)
    csv_string = csv_output.getvalue()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RECORD_DIR}/session_{timestamp}.csv"
    try:
        with open(filename, "w", newline='', encoding='utf-8') as f:
            f.write(csv_string)
        print(f"[INFO] Saved session record to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV to disk: {e}")

    transcript_lines = [f"[{e['time']}s] Patient: {e['text']}" for e in transcript_data]
    full_text_log = "\n".join(transcript_lines) or "(No speech detected)"
    safe_transcript = redact_pii(full_text_log)

    prompt = f"""
    You are an advanced Behavioral Session Analyzer AI.
    RAW TRANSCRIPT:
    {safe_transcript}
    
    BEHAVIORAL SUMMARY (Sensors):
    - Emotion: {sensor_summary['face_emotion']}
    - Clinical Flag: {sensor_summary['clinical_flag']}
    
    TASK: Generate two outputs.
    1. transcript_tagged: Rewrite transcript verbatim with behavioral notes.
    2. report_content: Professional analysis sections.
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
                report_data["report_content"] = response.text or "Error parsing AI response"
                report_data["transcript_tagged"] = "Format Error"
    except Exception as e:
        report_data["report_content"] = f"AI Generation Failed: {e}"
    
    if not report_data["report_content"] or "Error" in str(report_data["report_content"]):
        annotated, local_report = generate_local_annotations(safe_transcript)
        report_data["transcript_tagged"] = annotated
        report_data["report_content"] = local_report

    return JSONResponse(content=report_data)

# --- VIDEO LOGIC (SMOOTHED BLINK) ---
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): 
        return None
    
    for _ in range(5): cap.read()
    
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        # Gaze history (for looking away)
        gaze_history = deque(maxlen=GAZE_SMOOTHING)
        
        # --- NEW: Blink History (Stabilizer) ---
        # Stores last 3 frames of Eye Aspect Ratio to prevent flickering
        ear_history = deque(maxlen=3)
        
        calib_offset = (0.0, 0.0)
        calib_buffer = []
        calibrated = False
        read_failures = 0

        # UPDATED THRESHOLD: Slightly higher makes "Closed" easier to trigger
        SMOOTH_EAR_THRESHOLD = 0.21 

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    read_failures += 1
                    if read_failures > 10: break 
                    continue
                read_failures = 0 
                
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
                        
                        # 1. Calculate Raw EAR
                        left_ear = get_ear(lm, eye_idxs[0], w, h)
                        right_ear = get_ear(lm, eye_idxs[1], w, h)
                        current_ear = (left_ear + right_ear) / 2.0
                        
                        # 2. Smooth it (Average of last 3 frames)
                        ear_history.append(current_ear)
                        avg_ear = sum(ear_history) / len(ear_history)
                        
                        # 3. Check Blink based on Average
                        if avg_ear < SMOOTH_EAR_THRESHOLD:
                            # EYES CLOSED
                            status_text = "Eyes Closed"
                            state_manager.update("blink_state", "Closed")
                            state_manager.update("gaze_status", status_text)
                        else:
                            # EYES OPEN -> Calculate Gaze
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

                                # Draw Eyes
                                cv2.circle(frame, (int(l_iris_center[0]), int(l_iris_center[1])), 2, (0, 0, 255), -1)
                                cv2.circle(frame, (int(r_iris_center[0]), int(r_iris_center[1])), 2, (0, 0, 255), -1)
                            else:
                                state_manager.update("gaze_status", "Obstructed")

                    else:
                        state_manager.update("gaze_status", "No Face")
                        state_manager.update("face_emotion", "No Face")
                        state_manager.update("face_conf", 0.0)
                        state_manager.update("blink_state", "N/A")

                except Exception: pass
                
                # Overlay
                snap = state_manager.get_snapshot()
                cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Gaze: {snap['gaze_status']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {snap['face_emotion']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)