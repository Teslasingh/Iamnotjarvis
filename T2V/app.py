from pathlib import Path
from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DB_FILE = str(BASE_DIR / "clear_speak_history.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT id, timestamp, text FROM history ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "text": r[2]} for r in rows]

@app.route("/")
def index():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        return "index.html not found next to app.py", 500
    return html_path.read_text(encoding="utf-8")

@app.route("/history")
def api_history():
    return jsonify(get_history())

@app.route("/save", methods=["POST"])
def api_save():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"success": False}), 400
    
    conn = sqlite3.connect(DB_FILE)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO history (timestamp, text) VALUES (?, ?)", (ts, text))
    # Keep only last 10
    conn.execute("DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY id DESC LIMIT 10)")
    conn.commit()
    conn.close()
    return jsonify({"success": True, "history": get_history()})

@app.route("/delete/<int:hid>", methods=["DELETE"])
def api_delete(hid):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM history WHERE id = ?", (hid,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "history": get_history()})

@app.route("/clear", methods=["DELETE"])
def api_clear():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return jsonify({"success": True, "history": []})

if __name__ == "__main__":
    init_db()
    print("\n🚀 ClearSpeak is running!")
    print("   → Local:   http://127.0.0.1:5000")
    print("   → Network: http://YOUR_LOCAL_IP:5000  (use on phone, tablet, other PCs)")
    print("   Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)