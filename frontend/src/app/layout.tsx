// src/app/layout.tsx — root layout

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Header from "@/components/layout/Header";
import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: {
    default: "BankBot — Dual-Engine Banking Chatbot",
    template: "%s | BankBot",
  },
  description:
    "AI-powered banking assistant using DistilBERT intent classification and Flan-T5 RAG generation.",
  keywords: ["banking chatbot", "AI", "DistilBERT", "Flan-T5", "RAG"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <body>
        <Header />
        <div className="page-wrapper">{children}</div>
      </body>
    </html>
  );
}
