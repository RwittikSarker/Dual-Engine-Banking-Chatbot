// hooks/useChat.ts — stateful chat hook

"use client";

import { useState, useCallback } from "react";
import { sendMessage } from "@/lib/api";
import type { Message, ChatState } from "@/types/chat";

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

export function useChat(): ChatState & {
  submit: (query: string) => Promise<void>;
  clear: () => void;
} {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return;

    setError(null);
    setIsLoading(true);

    // Optimistically add the user message and a loading placeholder
    const userMsg: Message = {
      id: generateId(),
      role: "user",
      content: query.trim(),
      timestamp: new Date(),
    };
    const loadingId = generateId();
    const loadingMsg: Message = {
      id: loadingId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);

    try {
      const result = await sendMessage(query.trim());
      const assistantMsg: Message = {
        id: loadingId,
        role: "assistant",
        content: result.generated_answer,
        timestamp: new Date(),
        result,
        isLoading: false,
      };
      // Replace loading placeholder with real message
      setMessages((prev) =>
        prev.map((m) => (m.id === loadingId ? assistantMsg : m))
      );
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      setError(errorMsg);
      // Replace loading message with error message
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                ...m,
                content: `⚠️ Failed to connect to the API: ${errorMsg}. Make sure the FastAPI server is running on port 8000.`,
                isLoading: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const clear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isLoading, error, submit, clear };
}
