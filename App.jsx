import { useState, useCallback } from "react";

const LATIN = {
  robin:     "Erithacus rubecula",
  chaffinch: "Fringilla coelebs",
  great_tit: "Great Tit",
  unknown:   "Sound not recognized",
};

const DISPLAY = {
  robin:     "European Robin",
  chaffinch: "Common Chaffinch",
  great_tit: "Parus major",
  unknown:   "No match found",
};

const COLORS = {
  robin:     { bg: "#1a3a2a", bar: "#7ee787", text: "#7ee787" },
  chaffinch: { bg: "#1a2a3a", bar: "#79c0ff", text: "#79c0ff" },
  great_tit: { bg: "#3a2a1a", bar: "#ffa657", text: "#ffa657" },
  unknown:   { bg: "#2a2a2a", bar: "#8b949e", text: "#8b949e" },
};

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round"
      style={{ width: 20, height: 20, stroke: "#7ee787" }}>
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}

function ConfidenceBar({ species, value, isWinner }) {
  const color = COLORS[species] || COLORS.unknown;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
        <span style={{ color: "#e6edf3" }}>{DISPLAY[species] || species}</span>
        <span style={{ color: "#8b949e" }}>{value.toFixed(1)}%</span>
      </div>
      <div style={{ height: 4, background: "#21262d", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${value}%`,
          borderRadius: 2,
          background: isWinner ? color.bar : "#238636",
          transition: "width 0.8s ease",
        }}/>
      </div>
    </div>
  );
}

function TimelineRow({ window: w }) {
  const color = COLORS[w.prediction] || COLORS.unknown;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
      <div style={{ width: 70, fontSize: 10, color: "#8b949e", textAlign: "right", flexShrink: 0 }}>
        {w.start}s – {w.end}s
      </div>
      <div style={{
        flex: 1, height: 28, background: color.bg,
        borderRadius: 4, display: "flex", alignItems: "center",
        paddingLeft: 10, gap: 8,
      }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: color.bar, flexShrink: 0 }}/>
        <span style={{ fontSize: 11, color: color.text, fontWeight: 500 }}>
          {DISPLAY[w.prediction] || w.prediction}
        </span>
        <span style={{ fontSize: 10, color: "#8b949e", marginLeft: "auto", paddingRight: 10 }}>
          {w.confidence}%
        </span>
      </div>
    </div>
  );
}

function SpeciesSummary({ timeline }) {
  const counts = {};
  timeline.forEach(w => {
    counts[w.prediction] = (counts[w.prediction] || 0) + 1;
  });
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
      {Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([species, count]) => {
          const color = COLORS[species] || COLORS.unknown;
          return (
            <div key={species} style={{
              padding: "6px 12px",
              background: color.bg,
              border: `1px solid ${color.bar}33`,
              borderRadius: 20,
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: color.bar }}/>
              <span style={{ fontSize: 11, color: color.text }}>
                {DISPLAY[species]} — {count} window{count > 1 ? "s" : ""}
              </span>
            </div>
          );
        })}
    </div>
  );
}

function SpectrogramViewer({ spectrogram, windowSpecs, timeline }) {
  const [activeTab, setActiveTab]       = useState("full");
  const [currentWindow, setCurrentWindow] = useState(0);

  return (
    <div style={{
      background: "#161b22", border: "1px solid #21262d",
      borderRadius: 12, padding: 24, width: "100%",
      maxWidth: 640, marginBottom: 20,
    }}>
      <div style={{ fontSize: 11, color: "#8b949e", letterSpacing: 2, textTransform: "uppercase", marginBottom: 16 }}>
        Spectrogram Analysis
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        {["full", "window"].map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)} style={{
            padding: "6px 16px", borderRadius: 6, fontSize: 11,
            letterSpacing: 1, textTransform: "uppercase",
            cursor: "pointer", fontFamily: "monospace",
            border: "1px solid",
            borderColor: activeTab === tab ? "#238636" : "#30363d",
            background: activeTab === tab ? "#238636" : "transparent",
            color: activeTab === tab ? "#fff" : "#8b949e",
            transition: "all 0.2s",
          }}>
            {tab === "full" ? "Full recording" : "Per window"}
          </button>
        ))}
      </div>

      {/* Full spectrogram */}
      {activeTab === "full" && spectrogram && (
        <div>
          <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 8 }}>
            Full recording — mel spectrogram (0 Hz to 8kHz)
          </div>
          <img
            src={`data:image/png;base64,${spectrogram}`}
            alt="Full spectrogram"
            style={{ width: "100%", borderRadius: 8, display: "block" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
            <span style={{ fontSize: 10, color: "#8b949e" }}>0s</span>
            <span style={{ fontSize: 10, color: "#8b949e" }}>time →</span>
            <span style={{ fontSize: 10, color: "#8b949e" }}>
              {timeline[timeline.length - 1]?.end}s
            </span>
          </div>
        </div>
      )}

      {/* Per window spectrogram */}
      {activeTab === "window" && windowSpecs && (
        <div>
          <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 12 }}>
            Window {currentWindow + 1} of {windowSpecs.length}
            {" — "}
            <span style={{ color: COLORS[timeline[currentWindow]?.prediction]?.bar || "#8b949e" }}>
              {DISPLAY[timeline[currentWindow]?.prediction] || "Unknown"}
            </span>
            {" — "}
            {timeline[currentWindow]?.start}s to {timeline[currentWindow]?.end}s
            {" — "}
            confidence: {timeline[currentWindow]?.confidence}%
          </div>

          <img
            src={`data:image/png;base64,${windowSpecs[currentWindow]}`}
            alt={`Window ${currentWindow + 1}`}
            style={{ width: "100%", borderRadius: 8, display: "block", marginBottom: 12 }}
          />

          {/* Navigation */}
          <div style={{ display: "flex", gap: 8, justifyContent: "center", alignItems: "center" }}>
            <button
              onClick={() => setCurrentWindow(Math.max(0, currentWindow - 1))}
              disabled={currentWindow === 0}
              style={{
                padding: "6px 16px", borderRadius: 6, fontSize: 11,
                fontFamily: "monospace", border: "1px solid #30363d",
                background: "transparent", cursor: currentWindow === 0 ? "not-allowed" : "pointer",
                color: currentWindow === 0 ? "#484f58" : "#8b949e",
              }}
            >← Prev</button>

            {/* Window dots */}
            <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
              {windowSpecs.map((_, i) => (
                <div key={i} onClick={() => setCurrentWindow(i)} style={{
                  width: 8, height: 8, borderRadius: "50%", cursor: "pointer",
                  background: i === currentWindow
                    ? COLORS[timeline[i]?.prediction]?.bar || "#7ee787"
                    : "#30363d",
                  transition: "background 0.2s",
                }}/>
              ))}
            </div>

            <button
              onClick={() => setCurrentWindow(Math.min(windowSpecs.length - 1, currentWindow + 1))}
              disabled={currentWindow === windowSpecs.length - 1}
              style={{
                padding: "6px 16px", borderRadius: 6, fontSize: 11,
                fontFamily: "monospace", border: "1px solid #30363d",
                background: "transparent",
                cursor: currentWindow === windowSpecs.length - 1 ? "not-allowed" : "pointer",
                color: currentWindow === windowSpecs.length - 1 ? "#484f58" : "#8b949e",
              }}
            >Next →</button>
          </div>
        </div>
      )}

      {/* Color scale */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 16 }}>
        <span style={{ fontSize: 10, color: "#8b949e" }}>-50 dB</span>
        <div style={{
          flex: 1, height: 8, borderRadius: 4,
          background: "linear-gradient(to right, #000080, #800080, #ff4500, #ffff00)"
        }}/>
        <span style={{ fontSize: 10, color: "#8b949e" }}>0 dB</span>
      </div>
      <div style={{ fontSize: 10, color: "#484f58", textAlign: "center", marginTop: 4 }}>
        bright = loud · dark = quiet · y-axis = frequency · x-axis = time
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);

  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const classify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res  = await fetch("http://localhost:5000/predict", {
        method: "POST", body: formData,
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to Python server. Make sure predict.py is running.");
    } finally {
      setLoading(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const card = {
    background: "#161b22", border: "1px solid #21262d",
    borderRadius: 12, padding: 32, width: "100%",
    maxWidth: 640, marginBottom: 20,
  };

  const pipeline = ["Load audio","3s windows","Mel spectrogram","Flatten","Normalize","100 trees","Vote"];

  return (
    <div style={{
      minHeight: "100vh", background: "#0d1117", color: "#e6edf3",
      fontFamily: "monospace", padding: "40px 20px",
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 48 }}>
        <h1 style={{ fontSize: 40, fontWeight: 300, color: "#e6edf3", letterSpacing: -1, marginBottom: 8 }}>
          Bird <span style={{ color: "#7ee787", fontStyle: "italic" }}>Audio</span> Classifier
        </h1>
        <p style={{ fontSize: 11, color: "#8b949e", letterSpacing: 2, textTransform: "uppercase" }}>
          European species · Random Forest · 94% accuracy
        </p>
      </div>

      {/* Upload card */}
      <div style={card}>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          style={{
            border: `1px dashed ${dragging ? "#7ee787" : "#30363d"}`,
            borderRadius: 8, padding: "48px 24px", textAlign: "center",
            cursor: "pointer", transition: "all 0.2s", position: "relative",
            background: dragging ? "rgba(126,231,135,0.04)" : "transparent",
          }}
        >
          <input type="file" accept=".wav,.mp3"
            onChange={(e) => handleFile(e.target.files[0])}
            style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer", width: "100%", height: "100%" }}
          />
          <div style={{
            width: 48, height: 48, margin: "0 auto 16px",
            border: "1px solid #30363d", borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <UploadIcon />
          </div>
          <div style={{ fontSize: 13, color: "#8b949e", lineHeight: 1.6 }}>
            <span style={{ color: "#7ee787" }}>Choose a file</span> or drag it here<br/>
            .wav or .mp3 · at least 3 seconds
          </div>
        </div>

        {file && (
          <div style={{
            display: "flex", alignItems: "center", gap: 12,
            padding: "12px 16px", marginTop: 16,
            background: "rgba(126,231,135,0.06)",
            border: "1px solid rgba(126,231,135,0.2)", borderRadius: 8,
          }}>
            <div style={{ fontSize: 12, color: "#7ee787", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {file.name}
            </div>
            <div style={{ fontSize: 11, color: "#8b949e" }}>{formatSize(file.size)}</div>
          </div>
        )}

        <button onClick={classify} disabled={!file || loading} style={{
          width: "100%", padding: 14, marginTop: 16,
          background: !file || loading ? "#21262d" : "#238636",
          color: !file || loading ? "#484f58" : "#fff",
          border: "none", borderRadius: 8, fontFamily: "monospace",
          fontSize: 13, letterSpacing: 1, textTransform: "uppercase",
          cursor: !file || loading ? "not-allowed" : "pointer",
          transition: "background 0.2s",
        }}>
          {loading ? "Analyzing..." : "Identify Species"}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div style={card}>
          <div style={{ display: "flex", justifyContent: "center", color: "#8b949e", fontSize: 12, letterSpacing: 1, marginBottom: 20 }}>
            Extracting mel spectrogram features...
          </div>
          <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", justifyContent: "center", gap: 4 }}>
            {pipeline.map((step, i) => (
              <span key={step} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ fontSize: 10, color: "#8b949e", letterSpacing: 1, textTransform: "uppercase", padding: "4px 8px", border: "1px solid #21262d", borderRadius: 4 }}>
                  {step}
                </span>
                {i < pipeline.length - 1 && <span style={{ color: "#30363d" }}>›</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={card}>
          <div style={{ padding: 16, background: "rgba(248,81,73,0.08)", border: "1px solid rgba(248,81,73,0.3)", borderRadius: 8, color: "#f85149", fontSize: 12, textAlign: "center" }}>
            {error}
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={card}>
          <div style={{ fontSize: 11, color: "#8b949e", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>
            {result.rejected ? "No match found" : "Dominant species"}
          </div>
          <div style={{ fontSize: 32, fontWeight: 300, color: COLORS[result.prediction]?.bar || "#7ee787", marginBottom: 4 }}>
            {DISPLAY[result.prediction] || result.prediction}
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", fontStyle: "italic", marginBottom: 24 }}>
            {LATIN[result.prediction]}
          </div>

          {result.timeline && <SpeciesSummary timeline={result.timeline} />}

          <div style={{ fontSize: 11, color: "#8b949e", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>
            Average confidence
          </div>
          {Object.entries(result.confidence)
            .sort((a, b) => b[1] - a[1])
            .map(([sp, val]) => (
              <ConfidenceBar key={sp} species={sp} value={val} isWinner={sp === result.prediction}/>
            ))}

          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12,
            marginTop: 20, paddingTop: 20, borderTop: "1px solid #21262d", textAlign: "center",
          }}>
            {[
              { val: `${result.duration}s`, key: "Duration" },
              { val: result.windows,        key: "Windows"  },
              { val: `${result.votes[result.prediction] || 0}/${result.windows}`, key: "Votes won" },
            ].map(({ val, key }) => (
              <div key={key}>
                <div style={{ fontSize: 20, color: "#e6edf3", fontWeight: 500 }}>{val}</div>
                <div style={{ fontSize: 10, color: "#8b949e", letterSpacing: 1, textTransform: "uppercase", marginTop: 2 }}>{key}</div>
              </div>
            ))}
          </div>

          {result.timeline && result.timeline.length > 0 && (
            <>
              <div style={{ fontSize: 11, color: "#8b949e", letterSpacing: 2, textTransform: "uppercase", margin: "24px 0 16px" }}>
                Timeline — what happened when
              </div>
              {result.timeline.map((w, i) => (
                <TimelineRow key={i} window={w}/>
              ))}
            </>
          )}
        </div>
      )}

      {/* Spectrogram viewer */}
      {result && result.spectrogram && (
        <SpectrogramViewer
          spectrogram={result.spectrogram}
          windowSpecs={result.window_specs}
          timeline={result.timeline}
        />
      )}

      <div style={{ fontSize: 10, color: "#484f58", letterSpacing: 1, textAlign: "center" }}>
        DS FINAL PROJECT · RANDOM FOREST · 100 TREES · 465 CLIPS
      </div>
    </div>
  );
}