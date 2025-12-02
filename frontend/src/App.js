import React, { useState, useEffect, useRef } from 'react';
import { 
  AlertTriangle, Mic, Video, Activity, 
  Eye, EyeOff, User, MessageSquare, Monitor, Move,
  Play, Square, FileText, Download, X
} from 'lucide-react';
import './App.css';

// Configuration
const API_URL = "http://localhost:8000";
const API_KEY = "dev-secret"; // Must match backend .env

function App() {
  const [data, setData] = useState({
    status: "Connecting...",
    face_emotion: "Neutral",
    face_conf: 0.0,
    gaze_status: "N/A",
    blink_state: "Open",
    vocal_emotion: "N/A",
    text: "",
    sentiment: "Neutral",
    clinical_flag: "None",
    alert_active: false
  });

  const [history, setHistory] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('report'); // 'report' or 'transcript'

  const ws = useRef(null);
  const chatEndRef = useRef(null);

  // --- WEBSOCKET CONNECTION ---
  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws");
    
    ws.current.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        setData(parsed);

        // Append distinctive text to history
        if (parsed.text && (history.length === 0 || history[history.length-1].text !== parsed.text)) {
           const newEntry = {
             text: parsed.text,
             sentiment: parsed.sentiment,
             flag: parsed.clinical_flag,
             time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})
           };
           setHistory(prev => [...prev.slice(-50), newEntry]); 
        }
      } catch (e) {
        console.error("Websocket Parse Error:", e);
      }
    };

    return () => ws.current?.close();
  }, [history]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  // --- SESSION CONTROLS ---

  const startSession = async () => {
    try {
      await fetch(`${API_URL}/start_session`, {
        method: "POST",
        headers: { "x-api-key": API_KEY }
      });
      setIsRecording(true);
      setHistory([]); // Clear local chat
      setReportData(null);
    } catch (err) {
      alert("Failed to start session: " + err.message);
    }
  };

  const stopSession = async () => {
    setIsRecording(false);
    setIsLoading(true);
    try {
      const res = await fetch(`${API_URL}/stop_session`, {
        method: "POST",
        headers: { "x-api-key": API_KEY }
      });
      const json = await res.json();
      setReportData(json);
      setShowReport(true);
    } catch (err) {
      alert("Failed to generate report: " + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadCSV = () => {
    if (!reportData?.csv) return;
    const blob = new Blob([reportData.csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_log_${new Date().toISOString()}.csv`;
    a.click();
  };

  // --- UI HELPERS ---
  const getEmoColor = (emo) => {
    const map = { 
      'angry': '#ef4444', 'sad': '#3b82f6', 'happy': '#22c55e', 
      'fear': '#a855f7', 'surprise': '#eab308', 'neutral': '#94a3b8' 
    };
    return map[String(emo).toLowerCase()] || '#94a3b8';
  };

  return (
    <div className={`app-container ${data.alert_active ? 'alert-mode' : ''}`}>
      
      {/* --- LEFT SIDEBAR --- */}
      <aside className="sidebar">
        <div className="brand">
          <Activity className="brand-icon" />
          <div>
            <h1>TelePsych</h1>
            <span className="status-badge">System Online</span>
          </div>
        </div>

        {/* CONTROLS */}
        <div className="controls-wrapper">
          {!isRecording ? (
            <button className="btn btn-start" onClick={startSession}>
              <Play size={16} /> Start Session
            </button>
          ) : (
            <button className="btn btn-stop" onClick={stopSession}>
              <Square size={16} /> End & Generate Report
            </button>
          )}
        </div>

        <div className="metrics-wrapper">
          <div className="metric-tile" style={{borderColor: getEmoColor(data.face_emotion)}}>
            <div className="tile-header"><User size={18} /> <span>Facial Affect</span></div>
            <div className="tile-value" style={{color: getEmoColor(data.face_emotion)}}>{data.face_emotion}</div>
            <div className="confidence-bar">
              <div className="fill" style={{width: `${data.face_conf * 100}%`, background: getEmoColor(data.face_emotion)}}></div>
            </div>
          </div>

          <div className="metric-tile">
            <div className="tile-header"><Mic size={18} /> <span>Vocal Tone</span></div>
            <div className="tile-value text-purple">{data.vocal_emotion}</div>
          </div>

          <div className={`metric-tile ${data.gaze_status === "Looking Away" ? 'warn-tile' : ''}`}>
            <div className="tile-header">
              {data.gaze_status === "Looking at Screen" ? <Monitor size={18}/> : <Move size={18}/>}
              <span>Attention</span>
            </div>
            <div className="tile-value">{data.gaze_status}</div>
          </div>

          <div className={`metric-tile flag-tile ${data.clinical_flag !== "None" ? 'active' : ''}`}>
            <div className="tile-header"><AlertTriangle size={18} /> <span>Clinical Risk</span></div>
            <div className="tile-value">{data.clinical_flag}</div>
          </div>
        </div>
      </aside>

      {/* --- MAIN CONTENT --- */}
      <main className="main-view">
        <div className="video-section">
          <div className="video-frame">
            <img src={`${API_URL}/video_feed`} alt="Patient Stream" className="stream-img" />
            
            <div className="hud-layer">
              <div className="hud-status">
                <div className="hud-tag">
                  {data.blink_state === "Closed" ? <EyeOff size={14} color="#f87171"/> : <Eye size={14} color="#4ade80"/>} 
                  {data.blink_state}
                </div>
                <div className="hud-tag">
                  <Video size={14} color="#4ade80"/> Live 30fps
                </div>
                {isRecording && <div className="hud-tag rec-tag">● REC</div>}
              </div>
              
              {data.alert_active && (
                 <div className="video-alert-banner">
                   <AlertTriangle size={24} /> RISK DETECTED
                 </div>
              )}
            </div>
          </div>
        </div>

        <div className="transcript-section">
          <div className="section-title">
            <MessageSquare size={16} /> Live Transcript
          </div>
          <div className="chat-feed">
             {history.map((msg, i) => (
               <div key={i} className={`msg-row ${msg.sentiment}`}>
                 <span className="time">{msg.time}</span>
                 <div className="msg-bubble">
                   {msg.text}
                   {msg.flag !== "None" && <span className="flag-badge">{msg.flag}</span>}
                 </div>
               </div>
             ))}
             {data.text && (
               <div className="msg-row preview">
                 <span className="time">Now</span>
                 <div className="msg-bubble typing">{data.text}<span className="cursor">|</span></div>
               </div>
             )}
             <div ref={chatEndRef} />
          </div>
        </div>
      </main>

      {/* --- REPORT MODAL --- */}
      {showReport && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h2>Session Analysis Report</h2>
              <button className="close-btn" onClick={() => setShowReport(false)}><X size={20}/></button>
            </div>
            
            <div className="modal-tabs">
              <button className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`} onClick={() => setActiveTab('report')}>
                <Activity size={16} /> AI Summary
              </button>
              <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>
                <FileText size={16} /> Annotated Transcript
              </button>
              <button className="tab-btn download-btn" onClick={downloadCSV}>
                <Download size={16} /> Download CSV
              </button>
            </div>

            <div className="modal-body">
              {activeTab === 'report' ? (
                <div className="report-text">
                  <pre>{reportData?.report_content}</pre>
                </div>
              ) : (
                <div className="transcript-text">
                  <pre>{reportData?.transcript_tagged}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* --- LOADING OVERLAY --- */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Generating Behavioral Report...</p>
        </div>
      )}

    </div>
  );
}

export default App;