// lib/api.ts — thin wrapper around the FastAPI backend

import type { PredictResponse } from "@/types/chat";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function sendMessage(query: string): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json() as Promise<PredictResponse>;
}

export async function fetchExamples(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/examples`);
  if (!res.ok) throw new Error("Failed to load examples");
  const data = await res.json();
  return data.examples as string[];
}

export async function checkHealth(): Promise<{ status: string; pipeline_loaded: boolean }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}
