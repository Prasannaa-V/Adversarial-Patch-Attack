import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import { 
  ShieldAlert, 
  Unlock, 
  Lock, 
  Activity, 
  Video, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  RefreshCw,
  Sliders,
  Database
} from 'lucide-react';

const BACKEND_URL = 'http://localhost:3000';

function App() {
  const [gateStatus, setGateStatus] = useState('LOCKED');
  const [esp32Online, setEsp32Online] = useState(false);
  const [pipelineState, setPipelineState] = useState({
    fps: 0,
    sdnl_latency_ms: 0,
    active: false,
    lastUpdate: null
  });
  
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({
    totalAttacks: 0,
    avgConfidence: 0,
    pendingAck: 0
  });

  const [selectedAlert, setSelectedAlert] = useState(null);
  const [ackNotes, setAckNotes] = useState('');
  const [showOverrideConfirm, setShowOverrideConfirm] = useState(false);
  const [loading, setLoading] = useState(false);

  const socketRef = useRef(null);

  // Fetch initial data
  const fetchData = async () => {
    try {
      const resEvents = await fetch(`${BACKEND_URL}/api/events?limit=30`);
      const eventsData = await resEvents.json();
      setAlerts(eventsData.rows || []);

      const resStats = await fetch(`${BACKEND_URL}/api/stats`);
      const statsData = await resStats.json();
      setStats({
        totalAttacks: statsData.totalAttacks,
        avgConfidence: statsData.avgConfidence,
        pendingAck: statsData.pendingAck
      });
    } catch (err) {
      console.error('Error fetching baseline status:', err);
    }
  };

  useEffect(() => {
    fetchData();

    // Establish WebSocket Connection
    const socket = io(BACKEND_URL);
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Connected to Backend WebSocket');
    });

    socket.on('esp32_status', (data) => {
      setEsp32Online(data.online);
    });

    socket.on('pipeline_heartbeat', (data) => {
      setPipelineState({
        fps: data.fps,
        sdnl_latency_ms: data.sdnl_latency_ms,
        active: true,
        lastUpdate: Date.now()
      });
    });

    socket.on('gate_status_change', (data) => {
      setGateStatus(data.status);
    });

    socket.on('attack_alert', (newAlert) => {
      setAlerts(prev => [newAlert, ...prev]);
      // Play a subtle high-frequency alert sound
      try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
      } catch (e) {}

      // Refresh Stats
      fetchData();
    });

    socket.on('event_acked', (data) => {
      setAlerts(prev => 
        prev.map(item => 
          item.id === parseInt(data.id) 
            ? { ...item, operator_ack: 1, notes: data.notes } 
            : item
        )
      );
      fetchData();
    });

    // Check pipeline activity timeout (offline if no heartbeat for 10 seconds)
    const activeCheck = setInterval(() => {
      setPipelineState(prev => {
        if (prev.active && prev.lastUpdate && (Date.now() - prev.lastUpdate > 10000)) {
          return { ...prev, active: false, fps: 0, sdnl_latency_ms: 0 };
        }
        return prev;
      });
    }, 5000);

    return () => {
      socket.disconnect();
      clearInterval(activeCheck);
    };
  }, []);

  // Trigger manual gate override
  const handleGateOverride = async (action) => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/gate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setGateStatus(data.gate);
        setShowOverrideConfirm(false);
      }
    } catch (err) {
      console.error('Failed to trigger manual gate override:', err);
    } finally {
      setLoading(false);
    }
  };

  // Submit acknowledgment
  const handleAckSubmit = async (e) => {
    e.preventDefault();
    if (!selectedAlert) return;

    try {
      const res = await fetch(`${BACKEND_URL}/api/events/${selectedAlert.id}/ack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: ackNotes })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setSelectedAlert(null);
        setAckNotes('');
      }
    } catch (err) {
      console.error('Failed to submit event acknowledgment:', err);
    }
  };

  return (
    <>
      {/* HEADER */}
      <header className="app-header">
        <div className="logo-section">
          <ShieldAlert size={28} color="#6366f1" />
          <h1 className="logo-text">InfraEdge-Shield</h1>
        </div>
        
        <div className="system-status-badges">
          <div className={`status-badge ${pipelineState.active ? 'active' : ''}`}>
            <span className={`status-dot ${pipelineState.active ? 'green' : 'muted'}`} />
            Pipeline: {pipelineState.active ? `${pipelineState.fps.toFixed(1)} FPS` : 'Offline'}
          </div>
          <div className={`status-badge ${esp32Online ? 'active' : ''}`}>
            <span className={`status-dot ${esp32Online ? 'green' : 'muted'}`} />
            ESP32: {esp32Online ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {/* DASHBOARD BODY */}
      <div className="main-layout">
        
        {/* LEFT MAIN PANE */}
        <div className="content-pane">
          
          {/* STATS ROW */}
          <div className="stats-grid">
            <div className="glass-card">
              <div className="stat-item">
                <span className="stat-label">Total Attacks Blocked</span>
                <span className="stat-value" style={{ color: stats.totalAttacks > 0 ? '#ef4444' : '#fff' }}>
                  {stats.totalAttacks}
                </span>
              </div>
            </div>
            <div className="glass-card">
              <div className="stat-item">
                <span className="stat-label">Pending operator Acks</span>
                <span className="stat-value" style={{ color: stats.pendingAck > 0 ? '#f59e0b' : '#fff' }}>
                  {stats.pendingAck}
                </span>
              </div>
            </div>
            <div className="glass-card">
              <div className="stat-item">
                <span className="stat-label">Avg Attack Confidence</span>
                <span className="stat-value">
                  {stats.avgConfidence > 0 ? `${(stats.avgConfidence * 100).toFixed(1)}%` : '0%'}
                </span>
              </div>
            </div>
            <div className="glass-card">
              <div className="stat-item">
                <span className="stat-label">SDNL Latency</span>
                <span className="stat-value">
                  {pipelineState.active ? `${pipelineState.sdnl_latency_ms.toFixed(1)} ms` : '--'}
                </span>
              </div>
            </div>
          </div>

          {/* MAIN STREAM + ACTUATION GRID */}
          <div className="main-grid">
            
            {/* Live Camera Feed */}
            <div className="glass-card">
              <div className="card-title">
                <Video size={18} color="#6366f1" />
                Live Visual Feed
              </div>
              <div className="video-container">
                {pipelineState.active ? (
                  /* Simulating stream. In a real system, webcam or RTSP feed is active */
                  <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#070a13' }}>
                    <Activity size={48} className="pulse" color="#6366f1" />
                    <span style={{ fontSize: '14px', marginTop: '12px', color: '#94a3b8' }}>Pipeline streaming on edge host</span>
                    <div className="video-overlay-banner">
                      LIVE • {pipelineState.fps.toFixed(1)} FPS • PC CPU INFERENCE
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={48} color="#f59e0b" />
                    <span style={{ color: '#94a3b8', fontSize: '14px' }}>Camera feed offline — pipeline not running</span>
                  </div>
                )}
              </div>
            </div>

            {/* Actuation Gate Lock */}
            <div className="glass-card gate-control-card">
              <div className="card-title" style={{ width: '100%' }}>
                <Sliders size={18} color="#6366f1" />
                Gate Actuator Controls
              </div>

              <div className={`gate-badge-huge ${gateStatus === 'LOCKED' ? 'locked' : 'unlocked'}`}>
                {gateStatus === 'LOCKED' ? <Lock size={48} /> : <Unlock size={48} />}
              </div>

              <div style={{ textAlign: 'center', marginBottom: '10px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: '600' }}>
                  {gateStatus === 'LOCKED' ? 'GATE LOCKED' : 'GATE UNLOCKED'}
                </h3>
                <p style={{ fontSize: '12px', color: '#64748b', marginTop: '4px' }}>
                  State: {gateStatus} {esp32Online ? ' (ESP32 active)' : ' (ESP32 offline - failsafe active)'}
                </p>
              </div>

              {showOverrideConfirm ? (
                <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <p style={{ fontSize: '12px', color: '#f59e0b', textAlign: 'center', margin: '4px 0' }}>
                    Are you sure you want to {gateStatus === 'LOCKED' ? 'unlock' : 'lock'} the security gate?
                  </p>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button 
                      className={`btn ${gateStatus === 'LOCKED' ? 'btn-success' : 'btn-danger'}`}
                      onClick={() => handleGateOverride(gateStatus === 'LOCKED' ? 'unlock' : 'lock')}
                      disabled={loading}
                    >
                      Confirm
                    </button>
                    <button className="btn" style={{ background: '#1e293b' }} onClick={() => setShowOverrideConfirm(false)}>
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <button 
                  className={`btn ${gateStatus === 'LOCKED' ? 'btn-success' : 'btn-danger'}`}
                  onClick={() => setShowOverrideConfirm(true)}
                  disabled={!esp32Online}
                >
                  {gateStatus === 'LOCKED' ? 'Manual Gate Unlock' : 'Manual Gate Lock'}
                </button>
              )}
            </div>

          </div>

          {/* LOWER THUMBNAIL GALLERY */}
          <div className="glass-card">
            <div className="card-title">
              <Database size={18} color="#6366f1" />
              Recent Flagged Incidents
            </div>
            
            <div className="gallery-grid">
              {alerts.slice(0, 12).map((item) => (
                <div key={item.id} className="gallery-item" onClick={() => {
                  setSelectedAlert(item);
                  setAckNotes(item.notes || '');
                }}>
                  {item.frame_path ? (
                    <img 
                      src={`${BACKEND_URL}/frames/${item.frame_path.split('data/frames/')[1]}`} 
                      className="gallery-img" 
                      alt="Incident cropped" 
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.style.display = 'none';
                      }}
                    />
                  ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1a1f35' }}>
                      <AlertTriangle size={24} color="#ef4444" />
                    </div>
                  )}
                  <div className="gallery-meta">
                    <div>Conf: {(item.confidence * 100).toFixed(0)}%</div>
                    <div style={{ color: item.operator_ack ? '#10b981' : '#ef4444' }}>
                      {item.operator_ack ? 'Acked' : 'Unacked'}
                    </div>
                  </div>
                </div>
              ))}
              {alerts.length === 0 && (
                <div style={{ gridColumn: '1 / -1', padding: '30px', textAlign: 'center', color: '#64748b' }}>
                  No attack events logged. System clean.
                </div>
              )}
            </div>
          </div>

        </div>

        {/* RIGHT SIDEBAR FEED */}
        <div className="sidebar-pane">
          <div className="sidebar-header">
            <span>Critical Alert Feed</span>
            <span style={{ fontSize: '11px', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', padding: '2px 8px', borderRadius: '100px', fontWeight: '600' }}>
              {stats.pendingAck} Pending
            </span>
          </div>

          <div className="alert-feed-list">
            {alerts.filter(a => !a.operator_ack).map((item) => (
              <div 
                key={item.id} 
                className="alert-item unacked"
                onClick={() => {
                  setSelectedAlert(item);
                  setAckNotes(item.notes || '');
                }}
              >
                <div className="alert-item-header">
                  <span className="alert-conf">{(item.confidence * 100).toFixed(0)}% Match</span>
                  <span className="alert-time">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="alert-item-body">
                  Adversarial patch signature detected. Bounding box ROI extracted.
                  <div style={{ fontSize: '10px', color: '#64748b', marginTop: '6px', fontFamily: 'monospace' }}>
                    Frame ID: {item.frame_id}
                  </div>
                </div>
              </div>
            ))}

            {alerts.filter(a => !a.operator_ack).length === 0 && (
              <div style={{ padding: '40px 20px', textAlign: 'center', color: '#64748b' }}>
                <CheckCircle size={32} color="#10b981" style={{ margin: '0 auto 12px' }} />
                <p style={{ fontSize: '13px' }}>All incident reports acknowledged.</p>
              </div>
            )}
          </div>
        </div>

      </div>

      {/* DETAIL AND ACKNOWLEDGEMENT MODAL */}
      {selectedAlert && (
        <div className="modal-overlay" onClick={() => setSelectedAlert(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 style={{ fontSize: '16px', fontWeight: '600' }}>
                Incident Details — Incident #{selectedAlert.id}
              </h2>
              <button 
                style={{ background: 'none', border: 'none', color: '#94a3b8', fontSize: '20px', cursor: 'pointer' }}
                onClick={() => setSelectedAlert(null)}
              >
                &times;
              </button>
            </div>
            
            <div className="modal-body">
              <div className="modal-img-container">
                {selectedAlert.frame_path ? (
                  <img 
                    src={`${BACKEND_URL}/frames/${selectedAlert.frame_path.split('data/frames/')[1]}`} 
                    className="modal-img" 
                    alt="Captured adversarial patch" 
                    onError={(e) => {
                      e.target.src = '';
                      e.target.alt = "Error loading frame image";
                    }}
                  />
                ) : (
                  <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#080a13' }}>
                    <AlertTriangle size={36} color="#ef4444" />
                    <span style={{ fontSize: '12px', color: '#64748b', marginTop: '8px' }}>Frame file not available on local host</span>
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '13px', background: 'rgba(255, 255, 255, 0.01)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <div>
                  <span style={{ color: '#64748b' }}>Detected at:</span>
                  <div style={{ fontWeight: '500', marginTop: '2px' }}>{new Date(selectedAlert.timestamp).toLocaleString()}</div>
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>SDNL Match Confidence:</span>
                  <div style={{ fontWeight: '600', color: '#ef4444', marginTop: '2px' }}>
                    {(selectedAlert.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>Frame Identifier:</span>
                  <div style={{ fontWeight: '500', marginTop: '2px', fontFamily: 'monospace' }}>{selectedAlert.frame_id}</div>
                </div>
                <div>
                  <span style={{ color: '#64748b' }}>Gate lock state:</span>
                  <div style={{ fontWeight: '600', color: '#ef4444', marginTop: '2px' }}>🔒 LOCKED</div>
                </div>
              </div>

              <form onSubmit={handleAckSubmit} className="ack-form">
                <span style={{ fontSize: '13px', color: '#94a3b8' }}>Operator Acknowledgment Notes:</span>
                <textarea
                  className="textarea-custom"
                  placeholder="Enter classification notes or incident analysis remarks..."
                  value={ackNotes}
                  onChange={(e) => setAckNotes(e.target.value)}
                  disabled={selectedAlert.operator_ack}
                />
                
                {!selectedAlert.operator_ack ? (
                  <button type="submit" className="btn btn-primary" style={{ marginTop: '8px' }}>
                    Acknowledge Alert
                  </button>
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', background: 'rgba(16, 185, 129, 0.08)', padding: '10px', borderRadius: '8px', fontSize: '13px' }}>
                    <CheckCircle size={18} />
                    <span>Incident acknowledged by operator</span>
                  </div>
                )}
              </form>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
