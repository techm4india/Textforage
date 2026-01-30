// src/RagChat.js
import React, { useState } from "react";
import axios from "axios";

function RagChat() {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith(".txt")) {
      alert("Please upload a .txt file for now.");
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      const text = reader.result;

      try {
        setUploadStatus("Uploading & indexing...");
        const res = await axios.post("http://localhost:5000/api/upload-doc", {
          text,
        });
        setUploadStatus(`Indexed ${res.data.chunks} chunks ✅`);
      } catch (err) {
        console.error(err);
        setUploadStatus("Upload failed ❌");
      }
    };
    reader.readAsText(file);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setChat((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const res = await axios.post("http://localhost:5000/api/chat-rag", {
        message: userMessage.text,
      });

      const aiMessage = { sender: "ai", text: res.data.reply };
      setChat((prev) => [...prev, aiMessage]);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="page">
      <h1>📚 RAG Chatbot </h1>

      <div className="upload-section">
        <p>Upload a .txt file to use as knowledge base:</p>
        <input type="file" accept=".txt" onChange={handleFileChange} />
        <p>{uploadStatus}</p>
      </div>

      <div className="chat-box">
        {chat.map((msg, i) => (
          <div
            key={i}
            className={msg.sender === "user" ? "user-msg" : "ai-msg"}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          placeholder="Ask something from the uploaded document..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default RagChat;
