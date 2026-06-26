"use client";

// src/app/chat/page.tsx — Main chat interface

import { useEffect, useState } from "react";
import { useChat } from "@/hooks/useChat";
import { fetchExamples } from "@/lib/api";
import ChatWindow from "@/components/chat/ChatWindow";
import MessageInput from "@/components/chat/MessageInput";
import Sidebar from "@/components/chat/Sidebar";

export default function ChatPage() {
  const { messages, isLoading, submit, clear } = useChat();
  const [examples, setExamples] = useState<string[]>([]);

  useEffect(() => {
    fetchExamples()
      .then(setExamples)
      .catch(() =>
        setExamples([
          "My card was charged twice",
          "How do I cancel a direct debit?",
          "I forgot my passcode",
          "My card was stolen",
        ])
      );
  }, []);

  return (
    <div className="chat-layout">
      <Sidebar
        examples={examples}
        onSelect={submit}
        onClear={clear}
        messageCount={messages.length}
      />

      <div className="chat-main">
        <ChatWindow messages={messages} />
        <MessageInput
          onSubmit={submit}
          disabled={isLoading}
          placeholder="Ask a banking question… (Enter to send)"
        />
      </div>
    </div>
  );
}
