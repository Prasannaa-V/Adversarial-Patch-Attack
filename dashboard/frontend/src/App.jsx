import React, { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';
import {
  ShieldAlert, Unlock, Lock, Activity, AlertTriangle,
  CheckCircle, Zap, Cpu, Database, Radio, Eye,
  BarChart3, Clock, LogIn, LogOut, User,
} from 'lucide-react';

const BACKEND_URL = 'http://localhost:3000';
const TOKEN_KEY   = 'infraedge_jwt';

// ── Auth helpers ──────────────────────────────────────────────────────────────
const saveToken = (t) => localStorage.setItem(TOKEN_KEY, t);
const loadToken = ()  => localStorage.getItem(TOKEN_KEY);
const clearToken = () => localStorage.removeItem(TOKEN_KEY);
const authHeader = () => {
  const t = loadToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
};

// ── Login Screen ──────────────────────────────────────────────────────────────
function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      const res = await fetch(`${BACKEND_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const d = await res.json();
      if (!res.ok) { setError(d.error || 'Login failed'); return; }
      saveToken(d.token);
      onLogin(d.user);
    } catch {
      setError('Cannot reach backend — is the server running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-overlay">
      <div className="login-card">
        <div className="login-logo">
          <div className="logo-icon-wrap" style={{ width: 52, height: 52, borderRadius: 14 }}>
            <ShieldAlert size={26} color="#818cf8" />
          </div>
          <div className="logo-text" style={{ fontSize: 22 }}>InfraEdge-Shield</div>
          <div className="logo-subtitle" style={{ marginTop: 4 }}>Operator Authentication Required</div>
        </div>

        <form onSubmit={submit} className="login-form">
          <div className="login-field">
            <label className="ack-label">Username</label>
            <input
              className="textarea-custom"
              style={{ minHeight: 'auto', padding: '10px 12px', resize: 'none' }}
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoComplete="username"
              required
            />
          </div>
          <div className="login-field">
            <label className="ack-label">Password</label>
            <input
              className="textarea-custom"
              style={{ minHeight: 'auto', padding: '10px 12px', resize: 'none' }}
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoComplete="current-password"
              placeholder="password123"
              required
            />
          </div>

          {error && (
            <div style={{ fontSize: 12, color: 'var(--red-light)', background: 'var(--red-dim)', padding: '9px 12px', borderRadius: 7, border: '1px solid rgba(239,68,68,0.25)' }}>
              {error}
            </div>
          )}

          <button type="submit" className="btn btn-primary" disabled={loading}>
            <LogIn size={15} />
            {loading ? 'Authenticating…' : 'Sign In'}
          </button>
        </form>

        <p style={{ fontSize: 10, color: 'var(--text-muted)', textAlign: 'center', marginTop: 12 }}>
          JWT-secured · Zero-trust HMAC-SHA256
        </p>
      </div>
    </div>
  );
}

// ── Live Feed ─────────────────────────────────────────────────────────────────
function LiveFeed({ pipelineActive, fps }) {
  const [imgSrc, setImgSrc]     = useState(null);
  const [hasFrame, setHasFrame] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    let active = true;
    let timerId = null;

    const poll = async () => {
      if (!active) return;
      try {
        const url = `${BACKEND_URL}/api/live-frame?t=${Date.now()}`;
        const res = await fetch(url, { cache: 'no-store' });
        if (!active) return;
        if (!res.ok) {
          setHasFrame(false);
          timerId = setTimeout(poll, 300);
          return;
        }
        const blob = await res.blob();
        if (!active) return;
        const objectUrl = URL.createObjectURL(blob);
        setImgSrc(prev => { if (prev) URL.revokeObjectURL(prev); return objectUrl; });
        setHasFrame(true);
        timerId = setTimeout(poll, 60); // smooth ~15 FPS video playback
      } catch {
        if (active) {
          setHasFrame(false);
          timerId = setTimeout(poll, 300);
        }
      }
    };

    poll();
    return () => {
      active = false;
      if (timerId) clearTimeout(timerId);
    };
  }, []);

  if (!hasFrame || !imgSrc) {
    return (
      <div className="video-offline">
        <AlertTriangle size={40} color="#f59e0b" />
        <span className="video-offline-text">Camera feed offline — start the pipeline</span>
      </div>
    );
  }

  return hasFrame && imgSrc ? (
    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
      <img src={imgSrc} alt="Live annotated feed"
        style={{ maxWidth: '100%', maxHeight: '100%', width: 'auto', height: 'auto', objectFit: 'contain', display: 'block' }} />
      <div className="video-overlay-banner">LIVE · ANNOTATED · ONNX CPU</div>
      <div className="video-overlay-fps">{(fps || 15).toFixed(1)} FPS</div>
    </div>
  ) : (
    <div className="video-active-body">
      <div className="video-pulse-ring">
        <Activity size={34} color="#818cf8" />
      </div>
      <span style={{ fontSize: 13, marginTop: 14, color: 'var(--text-secondary)' }}>
        Pipeline active — awaiting first frame…
      </span>
      <div className="video-overlay-banner">LIVE · PC EDGE INFERENCE · ONNX CPU</div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
function App() {
  const [user, setUser]           = useState(null);
  const [gateStatus, setGateStatus] = useState('LOCKED');
  const [esp32Online, setEsp32Online] = useState(false);
  const [pipelineState, setPipelineState] = useState({
    fps: 0, sdnl_latency_ms: 0, active: false, lastUpdate: null,
  });
  const [alerts, setAlerts]   = useState([]);
  const [stats, setStats]     = useState({ totalAttacks: 0, avgConfidence: 0, pendingAck: 0, hourlyDistribution: [] });
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [ackNotes, setAckNotes]           = useState('');
  const [showOverrideConfirm, setShowOverrideConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [latestAlert, setLatestAlert] = useState(null);
  const socketRef = useRef(null);

  // Check if already logged in on mount
  useEffect(() => {
    const token = loadToken();
    if (token) {
      // Validate token by pinging a protected endpoint
      fetch(`${BACKEND_URL}/api/stats`, { headers: authHeader() })
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(() => setUser({ username: 'admin' }))  // token is valid
        .catch(() => clearToken());                   // token expired
    }
  }, []);

  const fetchData = useCallback(async () => {
    try {
      const [evRes, stRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/events?limit=30`),
        fetch(`${BACKEND_URL}/api/stats`),
      ]);
      const ev = await evRes.json();
      const st = await stRes.json();
      setAlerts(ev.rows || []);
      setStats({
        totalAttacks: st.totalAttacks,
        avgConfidence: st.avgConfidence,
        pendingAck: st.pendingAck,
        hourlyDistribution: st.hourlyDistribution || [],
      });
    } catch {}
  }, []);

  useEffect(() => {
    fetchData();
    const socket = io(BACKEND_URL);
    socketRef.current = socket;
    socket.on('esp32_status', d => setEsp32Online(d.online));
    socket.on('pipeline_heartbeat', d =>
      setPipelineState({ fps: d.fps, sdnl_latency_ms: d.sdnl_latency_ms, active: true, lastUpdate: Date.now() })
    );
    socket.on('gate_status_change', d => setGateStatus(d.status));
    socket.on('attack_alert', newAlert => {
      setAlerts(prev => [newAlert, ...prev]);
      setLatestAlert(newAlert);
      setTimeout(() => setLatestAlert(null), 6000);
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        osc.type = 'sine'; osc.frequency.setValueAtTime(880, ctx.currentTime);
        osc.connect(ctx.destination); osc.start(); osc.stop(ctx.currentTime + 0.25);
      } catch {}
      fetchData();
    });
    socket.on('event_acked', d => {
      setAlerts(prev => prev.map(a => a.id === parseInt(d.id) ? { ...a, operator_ack: 1, notes: d.notes } : a));
      fetchData();
    });
    const checker = setInterval(() => {
      setPipelineState(prev => {
        if (prev.active && prev.lastUpdate && Date.now() - prev.lastUpdate > 10000)
          return { ...prev, active: false, fps: 0, sdnl_latency_ms: 0 };
        return prev;
      });
    }, 5000);
    return () => { socket.disconnect(); clearInterval(checker); };
  }, [fetchData]);

  const handleGateOverride = async (action) => {
    const token = loadToken();
    if (!token) { alert('You must be logged in to override the gate.'); return; }
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/gate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader() },
        body: JSON.stringify({ action }),
      });
      const d = await res.json();
      if (res.status === 401 || res.status === 403) {
        alert('Session expired — please log in again.'); clearToken(); setUser(null); return;
      }
      if (d.status === 'ok') { setGateStatus(d.gate); setShowOverrideConfirm(false); }
    } catch {} finally { setLoading(false); }
  };

  const handleAckSubmit = async (e) => {
    e.preventDefault();
    if (!selectedAlert) return;
    const token = loadToken();
    if (!token) { alert('You must be logged in to acknowledge.'); return; }
    try {
      const res = await fetch(`${BACKEND_URL}/api/events/${selectedAlert.id}/ack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader() },
        body: JSON.stringify({ notes: ackNotes }),
      });
      if (res.status === 401 || res.status === 403) {
        alert('Session expired — please log in again.'); clearToken(); setUser(null); return;
      }
      const d = await res.json();
      if (d.status === 'ok') { setSelectedAlert(null); setAckNotes(''); }
    } catch {}
  };

  const unacked  = alerts.filter(a => !a.operator_ack);
  const isLocked = gateStatus === 'LOCKED';

  if (!user) return <LoginScreen onLogin={setUser} />;

  return (
    <>
      {/* ── HEADER ── */}
      <header className="app-header">
        <div className="logo-section">
          <div className="logo-icon-wrap">
            <ShieldAlert size={20} color="#818cf8" />
          </div>
          <div>
            <div className="logo-text">InfraEdge-Shield</div>
            <div className="logo-subtitle">CYBER-PHYSICAL ACCESS DEFENSE</div>
          </div>
        </div>

        <div className="header-right">
          <div className={`status-pill ${pipelineState.active ? 'online' : ''}`}>
            <span className={`status-dot ${pipelineState.active ? 'green' : 'muted'}`} />
            {pipelineState.active ? `Pipeline · ${pipelineState.fps.toFixed(1)} FPS` : 'Pipeline Offline'}
          </div>
          <div className={`status-pill ${esp32Online ? 'online' : ''}`}>
            <span className={`status-dot ${esp32Online ? 'green' : 'muted'}`} />
            {esp32Online ? 'ESP32 Online' : 'ESP32 Offline'}
          </div>
          <div className={`status-pill ${isLocked ? 'danger' : 'online'}`}>
            {isLocked ? <Lock size={12} /> : <Unlock size={12} />}
            Gate {gateStatus}
          </div>
          <div className="status-pill online" style={{ gap: 6 }}>
            <User size={12} />
            {user.username}
          </div>
          <button
            className="btn btn-ghost"
            style={{ width: 'auto', padding: '5px 11px', fontSize: 11 }}
            onClick={() => { clearToken(); setUser(null); }}
          >
            <LogOut size={12} /> Logout
          </button>
        </div>
      </header>

      {/* ── THREAT BANNER ── */}
      {latestAlert && (
        <div className="threat-banner">
          <span className="threat-banner-dot" />
          <span className="threat-banner-label">⚠ THREAT DETECTED</span>
          Adversarial patch identified &nbsp;·&nbsp; Confidence&nbsp;
          <strong>{(latestAlert.confidence * 100).toFixed(1)}%</strong>
          &nbsp;·&nbsp; Frame&nbsp;
          <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}>{latestAlert.frame_id}</code>
          &nbsp;·&nbsp; Gate → <strong style={{ color: '#fca5a5' }}>LOCKED</strong>
        </div>
      )}

      {/* ── MAIN LAYOUT ── */}
      <div className="main-layout">
        <div className="content-pane">

          {/* STATS */}
          <div className="stats-grid">
            <div className="stat-card danger">
              <div className="stat-top">
                <span className="stat-label">Total Attacks Blocked</span>
                <div className="stat-icon red"><AlertTriangle size={14} color="#ef4444" /></div>
              </div>
              <span className={`stat-value ${stats.totalAttacks > 0 ? 'red' : ''}`}>{stats.totalAttacks}</span>
              <span className="stat-sub">All-time incident count</span>
            </div>
            <div className="stat-card warning">
              <div className="stat-top">
                <span className="stat-label">Pending Acks</span>
                <div className="stat-icon amber"><Clock size={14} color="#f59e0b" /></div>
              </div>
              <span className={`stat-value ${stats.pendingAck > 0 ? 'amber' : 'green'}`}>{stats.pendingAck}</span>
              <span className="stat-sub">Unreviewed incidents</span>
            </div>
            <div className="stat-card">
              <div className="stat-top">
                <span className="stat-label">Avg Confidence</span>
                <div className="stat-icon indigo"><BarChart3 size={14} color="#818cf8" /></div>
              </div>
              <span className="stat-value">
                {stats.avgConfidence > 0 ? `${(stats.avgConfidence * 100).toFixed(1)}%` : '—'}
              </span>
              <span className="stat-sub">SDNL match score</span>
            </div>
            <div className={`stat-card ${pipelineState.active ? 'success' : ''}`}>
              <div className="stat-top">
                <span className="stat-label">SDNL Latency</span>
                <div className="stat-icon cyan"><Cpu size={14} color="#06b6d4" /></div>
              </div>
              <span className={`stat-value ${pipelineState.active ? 'cyan' : ''}`}>
                {pipelineState.active ? `${pipelineState.sdnl_latency_ms.toFixed(1)} ms` : '—'}
              </span>
              <span className="stat-sub">ONNX CPU · ≤15 ms target</span>
            </div>
          </div>

          {/* VIDEO + GATE */}
          <div className="main-grid">
            <div className="glass-card">
              <div className="card-title">
                <Eye size={16} color="#818cf8" />
                Live Visual Feed
                <span className="card-title-tag">{pipelineState.active ? '● ACTIVE' : '○ OFFLINE'}</span>
              </div>
              <div className="video-container">
                <LiveFeed pipelineActive={pipelineState.active} fps={pipelineState.fps} />
              </div>
            </div>

            <div className="glass-card gate-control-card">
              <div className="card-title" style={{ width: '100%' }}>
                <Radio size={14} color="#818cf8" />
                Gate Actuator
                <span className="card-title-tag">ESP32</span>
              </div>
              <div className={`gate-orb ${isLocked ? 'locked' : 'unlocked'}`}>
                {isLocked ? <Lock size={32} /> : <Unlock size={32} />}
              </div>
              <div className="gate-state-label" style={{ color: isLocked ? 'var(--red-light)' : 'var(--green-light)' }}>
                GATE {gateStatus}
              </div>
              <div className="gate-state-sub">
                {esp32Online ? 'Controller active' : 'Offline — fail-secure'}
              </div>
              <div className="gate-divider" />
              {showOverrideConfirm ? (
                <>
                  <p style={{ fontSize: 11, color: 'var(--amber)', textAlign: 'center' }}>
                    Confirm {isLocked ? 'unlock' : 'lock'}?
                  </p>
                  <div className="confirm-row">
                    <button
                      className={`btn ${isLocked ? 'btn-success' : 'btn-danger'}`}
                      onClick={() => handleGateOverride(isLocked ? 'unlock' : 'lock')}
                      disabled={loading}
                    >
                      {loading ? '…' : 'Confirm'}
                    </button>
                    <button className="btn btn-ghost" onClick={() => setShowOverrideConfirm(false)}>Cancel</button>
                  </div>
                </>
              ) : (
                <button
                  className={`btn ${isLocked ? 'btn-success' : 'btn-danger'}`}
                  onClick={() => setShowOverrideConfirm(true)}
                  disabled={!esp32Online}
                >
                  {isLocked ? <Unlock size={13} /> : <Lock size={13} />}
                  {isLocked ? 'Unlock Override' : 'Lock Override'}
                </button>
              )}
            </div>
          </div>

          {/* ANALYTICS — attacks by hour (today, UTC), no chart lib */}
          <div className="glass-card" style={{ marginTop: 14 }}>
            <div className="card-title">
              <BarChart3 size={14} color="#818cf8" />
              Attacks by Hour
              <span className="card-title-tag">TODAY UTC</span>
            </div>
            {(() => {
              const dist = stats.hourlyDistribution || [];
              if (dist.length === 0) {
                return <div className="stat-sub" style={{ padding: '4px 2px 8px' }}>No attacks logged today.</div>;
              }
              const max = Math.max(...dist.map(d => d.count), 1);
              return (
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 64, padding: '6px 2px 2px' }}>
                  {dist.map(d => (
                    <div key={d.hour} title={`${d.hour}:00 — ${d.count}`} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
                      <div style={{
                        width: '100%', borderRadius: 3,
                        height: `${Math.max(4, Math.round((d.count / max) * 48))}px`,
                        background: 'linear-gradient(180deg,#818cf8,#6366f1)',
                        opacity: 0.9,
                      }} />
                      <span style={{ fontSize: 8, color: 'var(--text-muted)' }}>{String(d.hour).padStart(2, '0')}</span>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>

          {/* GALLERY */}
          <div className="gallery-card">
            <div className="card-title">
              <Database size={14} color="#818cf8" />
              Flagged Incident Gallery
              <span className="card-title-tag">{alerts.length} EVENTS</span>
            </div>
            <div className="gallery-grid">
              {alerts.slice(0, 24).map(item => (
                <div
                  key={item.id}
                  className="gallery-item"
                  onClick={() => { setSelectedAlert(item); setAckNotes(item.notes || ''); }}
                >
                  {item.frame_path ? (
                    <img
                      src={`${BACKEND_URL}/frames/${item.frame_path.split('data/frames/')[1]}`}
                      className="gallery-img" alt="Incident"
                      onError={e => { e.target.style.display = 'none'; }}
                    />
                  ) : (
                    <div className="gallery-no-img"><AlertTriangle size={20} color="#ef4444" /></div>
                  )}
                  <div className="gallery-meta">
                    <span className="gallery-conf">{(item.confidence * 100).toFixed(0)}%</span>
                    <span className={`gallery-ack ${item.operator_ack ? 'done' : 'pending'}`} />
                  </div>
                </div>
              ))}
              {alerts.length === 0 && (
                <div className="empty-state">
                  <CheckCircle size={24} color="var(--green)" style={{ margin: '0 auto 8px' }} />
                  <div>No attack events logged. System clean.</div>
                </div>
              )}
            </div>
          </div>

        </div>

        {/* ── SIDEBAR ── */}
        <div className="sidebar-pane">
          <div className="sidebar-header">
            <span>Critical Alert Feed</span>
            <span className="sidebar-pending-badge">{stats.pendingAck} Pending</span>
          </div>
          <div className="alert-feed-list">
            {unacked.length === 0 ? (
              <div className="alert-all-clear">
                <CheckCircle size={30} color="var(--green)" />
                <span>All incidents acknowledged</span>
              </div>
            ) : (
              unacked.map(item => (
                <div
                  key={item.id}
                  className="alert-item unacked"
                  onClick={() => { setSelectedAlert(item); setAckNotes(item.notes || ''); }}
                >
                  <div className="alert-item-header">
                    <span className="alert-conf-badge">
                      <Zap size={10} style={{ marginRight: 3, verticalAlign: 'middle' }} />
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                    <span className="alert-time">{new Date(item.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <div className="alert-item-body">Adversarial patch signature detected.</div>
                  <div className="alert-frame-id">ID: {item.frame_id}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* ── INCIDENT MODAL ── */}
      {selectedAlert && (
        <div className="modal-overlay" onClick={() => setSelectedAlert(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>
                <AlertTriangle size={15} color="#ef4444" style={{ marginRight: 8, verticalAlign: 'middle' }} />
                Incident #{selectedAlert.id} — Review & Acknowledge
              </h2>
              <button className="modal-close" onClick={() => setSelectedAlert(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="modal-img-container">
                {selectedAlert.frame_path ? (
                  <img
                    src={`${BACKEND_URL}/frames/${selectedAlert.frame_path.split('data/frames/')[1]}`}
                    className="modal-img" alt="Adversarial patch"
                    onError={e => { e.target.alt = 'Frame not available'; }}
                  />
                ) : (
                  <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#030609' }}>
                    <AlertTriangle size={36} color="#ef4444" />
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10 }}>Frame snapshot not available</span>
                  </div>
                )}
              </div>
              <div className="modal-meta-grid">
                <div className="meta-item">
                  <label>Detected At</label>
                  <div className="meta-val">{new Date(selectedAlert.timestamp).toLocaleString()}</div>
                </div>
                <div className="meta-item">
                  <label>SDNL Confidence</label>
                  <div className="meta-val danger">{(selectedAlert.confidence * 100).toFixed(2)}%</div>
                </div>
                <div className="meta-item">
                  <label>Frame ID</label>
                  <div className="meta-val mono">{selectedAlert.frame_id}</div>
                </div>
                <div className="meta-item">
                  <label>Gate State</label>
                  <div className="meta-val danger">🔒 LOCKED (Fail-Secure)</div>
                </div>
              </div>
              <form onSubmit={handleAckSubmit} className="ack-form">
                <span className="ack-label">Operator Acknowledgment Notes</span>
                <textarea
                  className="textarea-custom"
                  placeholder="Enter classification notes or incident analysis…"
                  value={ackNotes}
                  onChange={e => setAckNotes(e.target.value)}
                  disabled={!!selectedAlert.operator_ack}
                />
                {!selectedAlert.operator_ack ? (
                  <button type="submit" className="btn btn-primary">
                    <CheckCircle size={15} /> Acknowledge Alert
                  </button>
                ) : (
                  <div className="ack-done-banner">
                    <CheckCircle size={18} /> Incident already acknowledged
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
