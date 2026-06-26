"use client";

// components/chat/MessageBubble.tsx

import type { Message } from "@/types/chat";
import MetricBadge from "./MetricBadge";

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  if (message.isLoading) {
    return (
      <div className="msg-row msg-row--assistant">
        <div className="avatar avatar--bot">🤖</div>
        <div className="bubble bubble--assistant">
          <div className="typing-indicator">
            <span /><span /><span />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--assistant"}`}>
      {!isUser && <div className="avatar avatar--bot">🤖</div>}

      <div className={`bubble ${isUser ? "bubble--user" : "bubble--assistant"}`}>
        <p className="bubble-text">{message.content}</p>

        {/* Metadata panel — only for assistant messages with a result */}
        {!isUser && message.result && (
          <div className="metadata">
            <div className="meta-row">
              <MetricBadge
                label="Intent"
                value={message.result.detected_intent.replace(/_/g, " ")}
                type="text"
              />
              <MetricBadge
                label="Confidence"
                value={`${(message.result.confidence_score * 100).toFixed(1)}%`}
                type={message.result.confidence_score >= 0.8 ? "high" : message.result.confidence_score >= 0.6 ? "med" : "low"}
              />
              <MetricBadge
                label="Similarity"
                value={`${(message.result.retrieval_similarity * 100).toFixed(1)}%`}
                type={message.result.retrieval_similarity >= 0.8 ? "high" : message.result.retrieval_similarity >= 0.5 ? "med" : "low"}
              />
            </div>

            {/* Escalation banner */}
            {message.result.escalation_recommendation === "Yes" ? (
              <div className="escalation escalation--warn">
                ⚠️ Escalation recommended — {message.result.escalation_reasons.join(", ")}
              </div>
            ) : (
              <div className="escalation escalation--ok">
                ✅ No escalation needed
              </div>
            )}

            {/* RAG raw output toggle */}
            {message.result.rag_answer && (
              <details className="rag-details">
                <summary className="rag-summary">🔬 RAG Engine (Flan-T5) raw output</summary>
                <p className="rag-text">{message.result.rag_answer}</p>
              </details>
            )}
          </div>
        )}

        <time className="bubble-time">
          {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </time>
      </div>

      {isUser && <div className="avatar avatar--user">👤</div>}
    </div>
  );
}
