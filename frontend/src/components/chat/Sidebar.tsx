"use client";

// components/chat/Sidebar.tsx — example queries & session actions

interface Props {
  examples: string[];
  onSelect: (q: string) => void;
  onClear: () => void;
  messageCount: number;
}

export default function Sidebar({ examples, onSelect, onClear, messageCount }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <h2 className="sidebar-heading">Dual-Engine Banking Chatbot</h2>
        <p className="sidebar-sub">
          Powered by DistilBERT + MiniLM + FAISS + Flan-T5
        </p>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <h3 className="sidebar-label">Try an example</h3>
        <ul className="example-list">
          {examples.map((q) => (
            <li key={q}>
              <button
                className="example-btn"
                onClick={() => onSelect(q)}
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <h3 className="sidebar-label">How it works</h3>
        <ol className="how-list">
          <li>Your message is classified by <strong>DistilBERT</strong></li>
          <li><strong>MiniLM + FAISS</strong> retrieves similar examples</li>
          <li><strong>Flan-T5</strong> generates a contextual answer</li>
          <li>The <strong>Escalation Engine</strong> safety-checks the result</li>
        </ol>
      </div>

      {messageCount > 0 && (
        <>
          <div className="sidebar-divider" />
          <div className="sidebar-section">
            <button className="clear-btn" onClick={onClear}>
              🗑 Clear conversation
            </button>
          </div>
        </>
      )}
    </aside>
  );
}
