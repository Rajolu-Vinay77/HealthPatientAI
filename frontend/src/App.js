import React, { useState, useEffect, useRef } from 'react';
import { 
  AlertTriangle, Heart, Mic, Video, Activity, 
  Eye, EyeOff, User, MessageSquare, Move, Monitor
} from 'lucide-react';
import './App.css';

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

  // Chat History State
  const [history, setHistory] = useState([]);
  const ws = useRef(null);
  const chatEndRef = useRef(null);

  // --- WEBSOCKET CONNECTION ---
  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws");
    
    ws.current.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      setData(parsed);

      // Append to history only if text is new and not empty
      if (parsed.text && (history.length === 0 || history[history.length-1].text !== parsed.text)) {
         const newEntry = {
           text: parsed.text,
           sentiment: parsed.sentiment,
           flag: parsed.clinical_flag,
           time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})
         };
         setHistory(prev => [...prev.slice(-50), newEntry]); // Keep last 50 msgs
      }
    };

    return () => ws.current?.close();
  }, [history]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  // Helper: Get Color based on Emotion
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

        <div className="metrics-wrapper">
          
          {/* 1. FACE CARD */}
          <div className="metric-tile" style={{borderColor: getEmoColor(data.face_emotion)}}>
            <div className="tile-header">
              <User size={18} /> <span>Facial Affect</span>
            </div>
            <div className="tile-value" style={{color: getEmoColor(data.face_emotion)}}>
              {data.face_emotion}
            </div>
            <div className="confidence-bar">
              <div className="fill" style={{width: `${data.face_conf * 100}%`, background: getEmoColor(data.face_emotion)}}></div>
            </div>
          </div>

          {/* 2. VOICE CARD */}
          <div className="metric-tile">
            <div className="tile-header">
              <Mic size={18} /> <span>Vocal Tone</span>
            </div>
            <div className="tile-value text-purple">
              {data.vocal_emotion}
            </div>
          </div>

          {/* 3. GAZE CARD */}
          <div className={`metric-tile ${data.gaze_status === "Looking Away" ? 'warn-tile' : ''}`}>
            <div className="tile-header">
              {data.gaze_status === "Looking at Screen" ? <Monitor size={18}/> : <Move size={18}/>}
              <span>Attention</span>
            </div>
            <div className="tile-value">
              {data.gaze_status}
            </div>
          </div>

          {/* 4. CLINICAL ALERT */}
          <div className={`metric-tile flag-tile ${data.clinical_flag !== "None" ? 'active' : ''}`}>
            <div className="tile-header">
              <AlertTriangle size={18} /> <span>Clinical Risk</span>
            </div>
            <div className="tile-value">
              {data.clinical_flag}
            </div>
          </div>

        </div>
      </aside>

      {/* --- MAIN CONTENT --- */}
      <main className="main-view">
        
        {/* VIDEO FEED */}
        <div className="video-section">
          <div className="video-frame">
            <img src="http://localhost:8000/video_feed" alt="Patient Stream" className="stream-img" />
            
            {/* HUD OVERLAY (The "High Tech" Look) */}
            <div className="hud-layer">
              <div className="hud-corners"></div>
              <div className="hud-status">
                <div className="hud-tag">
                  {data.blink_state === "Closed" ? <EyeOff size={14} color="#f87171"/> : <Eye size={14} color="#4ade80"/>} 
                  {data.blink_state}
                </div>
                <div className="hud-tag">
                  <Video size={14} color="#4ade80"/> Live 30fps
                </div>
              </div>
              
              {/* CRITICAL ALERT BANNER */}
              {data.alert_active && (
                 <div className="video-alert-banner">
                   <AlertTriangle size={24} /> RISK DETECTED
                 </div>
              )}
            </div>
          </div>
        </div>

        {/* TRANSCRIPT LOG */}
        <div className="transcript-section">
          <div className="section-title">
            <MessageSquare size={16} /> Live Transcript
          </div>
          <div className="chat-feed">
             {history.length === 0 && <div className="empty-msg">Waiting for speech...</div>}
             
             {history.map((msg, i) => (
               <div key={i} className={`msg-row ${msg.sentiment}`}>
                 <span className="time">{msg.time}</span>
                 <div className="msg-bubble">
                   {msg.text}
                   {msg.flag !== "None" && <span className="flag-badge">{msg.flag}</span>}
                 </div>
               </div>
             ))}
             
             {/* Live Typing Preview */}
             {data.text && (
               <div className="msg-row preview">
                 <span className="time">Now</span>
                 <div className="msg-bubble typing">
                   {data.text}<span className="cursor">|</span>
                 </div>
               </div>
             )}
             <div ref={chatEndRef} />
          </div>
        </div>

      </main>
    </div>
  );
}

export default App;