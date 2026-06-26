// components/layout/Header.tsx

import Link from "next/link";

export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <Link href="/" className="logo">
          <span className="logo-icon">🏦</span>
          <span className="logo-text">BankBot</span>
        </Link>
        <nav className="nav">
          <Link href="/chat" className="nav-link">Chat</Link>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link"
          >
            API Docs
          </a>
        </nav>
      </div>
    </header>
  );
}
