// src/NormalChat.js
import React, { useState } from "react";
import axios from "axios";

function NormalChat() {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setChat((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const res = await axios.post("http://localhost:5000/api/chat", {
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
      <h1>🤖 Chatbot </h1>

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
          placeholder="Type your message..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default NormalChat;
