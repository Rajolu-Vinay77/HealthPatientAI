import React, { useState, useEffect, useRef } from 'react';
import { 
  AlertTriangle, Mic, Video, Activity, 
  Eye, EyeOff, User, MessageSquare, Monitor, Move
} from 'lucide-react';
import './App.css';

function App() {
  // 1. State Management matches Backend 'SharedState' structure
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
  const ws = useRef(null);
  const chatEndRef = useRef(null);

  // 2. WebSocket Connection Logic
  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws");
    
    ws.current.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        setData(parsed);

        // 3. Chat History Logic: Only add distinctive, final text segments
        // We check if the text is different from the last entry to avoid duplicates
        if (parsed.text && (history.length === 0 || history[history.length-1].text !== parsed.text)) {
           const newEntry = {
             text: parsed.text,
             sentiment: parsed.sentiment,
             flag: parsed.clinical_flag,
             time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})
           };
           // Keep history manageable (last 50 entries)
           setHistory(prev => [...prev.slice(-50), newEntry]); 
        }
      } catch (e) {
        console.error("Websocket Parse Error:", e);
      }
    };

    ws.current.onclose = () => console.log("WebSocket Disconnected");
    return () => ws.current?.close();
  }, [history]); // Dependency on history ensures distinct check works correctly

  // 4. Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  // 5. Helper for Dynamic UI Colors
  const getEmoColor = (emo) => {
    const map = { 
      'angry': '#ef4444', 'sad': '#3b82f6', 'happy': '#22c55e', 
      'fear': '#a855f7', 'surprise': '#eab308', 'neutral': '#94a3b8' 
    };
    return map[String(emo).toLowerCase()] || '#94a3b8';
  };

  return (
    <div className={`app-container ${data.alert_active ? 'alert-mode' : ''}`}>
      
      {/* --- LEFT SIDEBAR (METRICS) --- */}
      <aside className="sidebar">
        <div className="brand">
          <Activity className="brand-icon" />
          <div>
            <h1>TelePsych</h1>
            <span className="status-badge">System Online</span>
          </div>
        </div>

        <div className="metrics-wrapper">
          
          {/* Metric 1: Facial Affect */}
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

          {/* Metric 2: Vocal Tone */}
          <div className="metric-tile">
            <div className="tile-header">
              <Mic size={18} /> <span>Vocal Tone</span>
            </div>
            <div className="tile-value text-purple">
              {data.vocal_emotion}
            </div>
          </div>

          {/* Metric 3: Attention / Gaze */}
          <div className={`metric-tile ${data.gaze_status === "Looking Away" ? 'warn-tile' : ''}`}>
            <div className="tile-header">
              {data.gaze_status === "Looking at Screen" ? <Monitor size={18}/> : <Move size={18}/>}
              <span>Attention</span>
            </div>
            <div className="tile-value">
              {data.gaze_status}
            </div>
          </div>

          {/* Metric 4: Clinical Risk Flag */}
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

      {/* --- MAIN CONTENT AREA --- */}
      <main className="main-view">
        
        {/* Video Stream Section */}
        <div className="video-section">
          <div className="video-frame">
            {/* The Source is your Python Backend URL */}
            <img src="http://localhost:8000/video_feed" alt="Patient Stream" className="stream-img" />
            
            {/* Heads-Up Display (HUD) Overlay */}
            <div className="hud-layer">
              <div className="hud-corners"></div>
              <div className="hud-status">
                <div className="hud-tag">
                  {/* Blink Indicator */}
                  {data.blink_state === "Closed" ? <EyeOff size={14} color="#f87171"/> : <Eye size={14} color="#4ade80"/>} 
                  {data.blink_state}
                </div>
                <div className="hud-tag">
                  <Video size={14} color="#4ade80"/> Live 30fps
                </div>
              </div>
              
              {/* Critical Alert Pop-up */}
              {data.alert_active && (
                 <div className="video-alert-banner">
                   <AlertTriangle size={24} /> RISK DETECTED
                 </div>
              )}
            </div>
          </div>
        </div>

        {/* Live Transcript Section */}
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
             
             {/* Real-time typing preview from partial results */}
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