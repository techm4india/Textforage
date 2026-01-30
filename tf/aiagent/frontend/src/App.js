import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  const testBackend = async () => {
    try {
      const res = await axios.get("http://localhost:5000/health");
      setHealth(JSON.stringify(res.data));
    } catch (err) {
      console.error("Health check error:", err);
      setHealth("❌ Cannot reach backend /health");
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const msgToSend = input;
    const userMessage = { sender: "user", text: msgToSend };

    setChat((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/api/chat", {
        message: msgToSend,
      });

      const aiMessage = { sender: "ai", text: res.data.reply };
      setChat((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error("Frontend /api/chat error:", err);
      const aiMessage = {
        sender: "ai",
        text: "❌ Backend or Python model not responding.",
      };
      setChat((prev) => [...prev, aiMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="app">
      <h1>🧠 AI Chat Agent</h1>

      <button onClick={testBackend} style={{ marginBottom: "10px" }}>
        🔍 Test Node Backend
      </button>
      {health && (
        <div style={{ marginBottom: "10px", fontSize: "14px" }}>
          Node health: {health}
        </div>
      )}

      <div className="chat-box">
        {chat.map((msg, i) => (
          <div
            key={i}
            className={msg.sender === "user" ? "user-msg" : "ai-msg"}
          >
            {msg.text}
          </div>
        ))}
        {loading && <div className="ai-msg">⏳ Generating...</div>}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          placeholder="Type your message..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default App;
