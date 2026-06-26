// src/app/page.tsx — Landing page

import Link from "next/link";

export const metadata = {
  title: "BankBot — Dual-Engine Banking Chatbot",
  description:
    "AI-powered banking assistant using DistilBERT intent classification and Flan-T5 RAG generation.",
};

export default function HomePage() {
  return (
    <main className="landing">
      <div className="landing-card">
        <div className="landing-icon">🏦</div>
        <h1 className="landing-title">Dual-Engine Banking Chatbot</h1>
        <p className="landing-desc">
          An AI assistant that classifies your intent with{" "}
          <strong>DistilBERT</strong>, retrieves context via{" "}
          <strong>MiniLM + FAISS</strong>, and generates answers with{" "}
          <strong>Flan-T5</strong>.
        </p>

        <div className="landing-pills">
          <span className="pill">DistilBERT</span>
          <span className="pill">MiniLM</span>
          <span className="pill">FAISS</span>
          <span className="pill">Flan-T5</span>
          <span className="pill">77 Intents</span>
        </div>

        <Link href="/chat" className="cta-btn">
          Start Chatting →
        </Link>

        <div className="landing-features">
          <div className="feature">
            <span className="feature-icon">🧠</span>
            <span className="feature-text">Intent Classification</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🔍</span>
            <span className="feature-text">Semantic Retrieval</span>
          </div>
          <div className="feature">
            <span className="feature-icon">⚡</span>
            <span className="feature-text">GPU Accelerated</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🛡️</span>
            <span className="feature-text">Escalation Engine</span>
          </div>
        </div>
      </div>
    </main>
  );
}
