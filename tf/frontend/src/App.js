// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import NormalChat from "./NormalChat";
import RagChat from "./RagChat";
import "./App.css";

function App() {
  return (
    <Router>
      <div className="app">
        {/* Simple navbar */}
        <nav className="navbar">
          <Link to="/">Normal Chatbot</Link>
          <Link to="/rag">RAG Chatbot</Link>
        </nav>

        <Routes>
          <Route path="/" element={<NormalChat />} />
          <Route path="/rag" element={<RagChat />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
