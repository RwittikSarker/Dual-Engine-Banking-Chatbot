// Types for the banking chatbot frontend

export type EscalationRecommendation = "Yes" | "No";

export interface PredictResponse {
  generated_answer: string;
  rag_answer: string;
  detected_intent: string;
  confidence_score: number;
  retrieval_similarity: number;
  escalation_recommendation: EscalationRecommendation;
  escalation_reasons: string[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  result?: PredictResponse;
  isLoading?: boolean;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}
