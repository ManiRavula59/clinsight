"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, Activity, Brain, User, Sparkles, BarChart2, Plus, Network, X, Database, ShieldCheck, Microscope, ThumbsUp, ThumbsDown, CheckCircle2, FlaskConical, Paperclip } from "lucide-react";
import { BackgroundDiffusion } from "@/components/ui/BackgroundDiffusion";
import { SquircleCard } from "@/components/ui/SquircleCard";
import { ArchitectureModal } from "@/components/ui/ArchitectureModal";
import { GenieModal } from "@/components/ui/GenieModal";
import { cn } from "@/lib/utils";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  traces?: string[];
  cases?: string[];
  isError?: boolean;
  file?: { type: string; base64: string };
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isWorkflowModalOpen, setIsWorkflowModalOpen] = useState(false);
  const [isMetricsModalOpen, setIsMetricsModalOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const metricsBtnRef = useRef<HTMLButtonElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startNewChat = () => {
    setMessages([]);
    setInput("");
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && !selectedFile) || isLoading) return;

    let fileData = undefined;
    if (selectedFile) {
      const base64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve((reader.result as string).split(',')[1]);
        reader.readAsDataURL(selectedFile);
      });
      fileData = { type: selectedFile.type, base64 };
    }

    const userMsg: Message = { 
      id: Date.now().toString(), 
      role: "user", 
      content: input || "Uploaded document for analysis.",
      file: fileData
    };
    
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setSelectedFile(null);
    setIsLoading(true);

    const assistantMsgId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, { id: assistantMsgId, role: "assistant", content: "", traces: [] }]);

    const messagesToSend = [...messages, userMsg].map(m => ({ 
      role: m.role, 
      content: m.content,
      ...(m.file && { file: m.file }) 
    }));

    try {
      // Setup SSE connection to our FastAPI backend
      const response = await fetch("http://localhost:8000/api/v1/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: messagesToSend }),
      });

      if (!response.ok) throw new Error("Network response error");
      if (!response.body) throw new Error("No body in response");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || ""; // Keep the last incomplete chunk in the buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "trace") {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, traces: [...(msg.traces || []), data.content] }
                    : msg
                ));
              } else if (data.type === "cases") {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, cases: data.data }
                    : msg
                ));
              } else if (data.type === "chunk") {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, content: msg.content + data.content }
                    : msg
                ));
              } else if (data.type === "error") {
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMsgId
                    ? { ...msg, content: msg.content + "\n\n**Error:** " + data.content, isError: true }
                    : msg
                ));
              }
            } catch (e) {
              console.error("Error parsing SSE JSON:", e, line);
            }
          }
        }
      }
    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMsgId
          ? { ...msg, content: msg.content + "\n\n**Connection Error:** " + String(error), isError: true }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen bg-white dark:bg-black font-sans selection:bg-cyan-100 dark:selection:bg-cyan-900 transition-colors duration-500">
      <BackgroundDiffusion />

      {/* Sidebar for History / Navigation (Simplified) */}
      <div className="fixed left-0 top-0 bottom-0 w-64 border-r border-black/5 dark:border-white/5 bg-white/50 dark:bg-black/50 backdrop-blur-3xl z-40 hidden md:flex flex-col p-4">
        <div className="flex items-center gap-2 mb-8 mt-2 px-2">
          <Activity className="w-6 h-6 text-cyan-500" />
          <span className="font-semibold text-lg tracking-tight">Clinsight</span>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2 mb-6 px-1">
          <button
            onClick={startNewChat}
            className="flex-1 flex items-center justify-center gap-2 bg-black dark:bg-white text-white dark:text-black py-2.5 px-3 rounded-xl font-medium text-sm hover:scale-[1.02] active:scale-[0.98] transition-transform shadow-sm"
          >
            <Plus className="w-4 h-4" />
            New Case
          </button>

          <button
            onClick={() => setIsWorkflowModalOpen(true)}
            className="flex items-center justify-center bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white p-2.5 rounded-xl hover:bg-gray-200 dark:hover:bg-white/20 transition-colors tooltip-trigger relative group"
            title="System Architecture"
          >
            <Network className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            <span className="absolute -top-10 left-1/2 -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
              View Architecture
            </span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="text-xs font-medium text-gray-400 mb-4 px-2 tracking-wider uppercase">History</div>
          {/* Mock history items */}
          <div className="px-2 py-2 rounded-xl text-sm hover:bg-black/5 dark:hover:bg-white/5 cursor-pointer transition-colors mb-1 truncate text-gray-700 dark:text-gray-300">
            Diabetic patient with chest pain
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="md:ml-64 flex-1 flex flex-col h-screen relative z-10">

        {/* Chat Area with massive bottom scroll padding so users can scroll past long LLM responses */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 md:space-y-8 scroll-smooth pb-[40vh]">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto opacity-0 animate-in fade-in zoom-in duration-1000 fill-mode-forwards">
              <div className="w-16 h-16 rounded-3xl bg-gradient-to-tr from-cyan-400 to-indigo-500 flex items-center justify-center mb-6 shadow-2xl shadow-cyan-500/20">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-3xl font-semibold mb-3 tracking-tight">How can I help you today?</h1>
              <p className="text-gray-500 dark:text-gray-400 text-lg mb-8">Enter a clinical case to search, or upload a prescription for AI safety verification.</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-xl text-left">
                <button 
                  onClick={() => setInput("Patient is a 55yo male. Allergies: Penicillin. History: Asthma. Currently taking: Albuterol inhaler. New Prescription: Amoxicillin 500mg 2x daily. Please setup English follow-up.")}
                  className="p-4 rounded-2xl bg-white dark:bg-white/5 border border-black/5 dark:border-white/10 hover:border-indigo-500/30 hover:bg-indigo-50/50 dark:hover:bg-indigo-500/10 transition-all group"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <ShieldCheck className="w-4 h-4 text-indigo-500" />
                    <span className="font-semibold text-sm">Prescription Engine (Allergy Conflict)</span>
                  </div>
                  <p className="text-xs text-gray-500">Test the Safety Guardian with an allergic reaction scenario.</p>
                </button>

                <button 
                  onClick={() => setInput("Patient Name: Sarah. Allergies: None. History: Mild Hypertension. New Prescription: Metoprolol 50mg 1x daily. Please setup English follow-up call.")}
                  className="p-4 rounded-2xl bg-white dark:bg-white/5 border border-black/5 dark:border-white/10 hover:border-cyan-500/30 hover:bg-cyan-50/50 dark:hover:bg-cyan-500/10 transition-all group"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="w-4 h-4 text-cyan-500" />
                    <span className="font-semibold text-sm">Prescription Engine (Safe + Twilio)</span>
                  </div>
                  <p className="text-xs text-gray-500">Test a safe prescription and trigger the AI Voice Follow-up.</p>
                </button>

                <button 
                  onClick={() => setInput("45-year-old male presenting with sudden crushing chest pain radiating to the left arm. Diaphoretic. History of smoking.")}
                  className="p-4 rounded-2xl bg-white dark:bg-white/5 border border-black/5 dark:border-white/10 hover:border-purple-500/30 hover:bg-purple-50/50 dark:hover:bg-purple-500/10 transition-all group"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-4 h-4 text-purple-500" />
                    <span className="font-semibold text-sm">RAG Case Retrieval Engine</span>
                  </div>
                  <p className="text-xs text-gray-500">Search 250,000 cases for similar chest pain diagnoses.</p>
                </button>

                <button 
                  onClick={() => setInput("60-year-old female with severe shortness of breath, bilateral leg swelling, and orthopnea. BP 160/90.")}
                  className="p-4 rounded-2xl bg-white dark:bg-white/5 border border-black/5 dark:border-white/10 hover:border-pink-500/30 hover:bg-pink-50/50 dark:hover:bg-pink-500/10 transition-all group"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Microscope className="w-4 h-4 text-pink-500" />
                    <span className="font-semibold text-sm">Complex Differential</span>
                  </div>
                  <p className="text-xs text-gray-500">Let the FAISS + Cross-Encoder identify the closest conditions.</p>
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={cn(
                "max-w-4xl mx-auto flex w-full",
                msg.role === "user" ? "justify-end" : "justify-start"
              )}>

                {msg.role === "assistant" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-black dark:bg-white text-white dark:text-black flex items-center justify-center mr-4 mt-2">
                    <Brain className="w-4 h-4" />
                  </div>
                )}

                <SquircleCard
                  dynamicScaling={!isLoading} // Restored dynamic hover, but strictly disabled during live generation to prevent parsing jitter
                  className={cn(
                    "max-w-[85%] relative overflow-hidden",
                    msg.role === "user"
                      ? "bg-gray-100 dark:bg-white/10 text-black dark:text-white ml-12"
                      : "bg-white/90 dark:bg-black/90 mr-12"
                  )}
                >
                  {/* Traces mapping strictly with conditional checks to avoid map crashes */}
                  {msg.role === "assistant" && msg.traces && msg.traces.length > 0 && (
                    <div className="mb-4 space-y-1">
                      {msg.traces.map((trace, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-xs text-gray-400 font-mono">
                          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                          {trace}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Render the Raw CrossEncoder Retrieved Cases in a Beautiful Accordion */}
                  {msg.role === "assistant" && msg.cases && msg.cases.length > 0 && (
                    <div className="mb-8 overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-500/5 to-cyan-500/5 border border-indigo-500/10 shadow-sm">
                      <div className="p-4 bg-indigo-500/5 border-b border-indigo-500/10 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-indigo-500" />
                        <span className="text-sm font-semibold text-indigo-700 dark:text-indigo-300">
                          {msg.cases.length} Evidence-Based Cases Retrieved
                        </span>
                      </div>
                      <div className="p-4 space-y-4 max-h-[500px] overflow-y-auto custom-scrollbar">
                        {msg.cases.map((clinicalCase, idx) => (
                          <details key={idx} className="group bg-white dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/10 shadow-sm overflow-hidden">
                            <summary className="p-4 cursor-pointer text-[15px] font-semibold text-gray-800 dark:text-gray-200 flex items-center list-none hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
                              <span className="flex items-center gap-3 flex-1">
                                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 text-xs font-bold">
                                  {idx + 1}
                                </span>
                                Source Document {idx + 1}
                              </span>
                              <span className="text-gray-400 group-open:rotate-180 transition-transform duration-300">▼</span>
                            </summary>
                            <div className="p-5 pt-4 text-[15px] text-gray-800 dark:text-gray-200 leading-relaxed border-t border-black/5 dark:border-white/10 bg-gray-50 dark:bg-white/5 whitespace-pre-wrap">
                              {clinicalCase.replace("Clinical Case:", "").trim()}
                            </div>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.role === "assistant" && msg.content ? (
                    <div className="space-y-6">
                      {msg.content.split(/(?=## )/).filter(Boolean).map((section, idx) => {
                        if (!section.trim().startsWith("## ")) {
                          // If it's just raw text before any header or missing a header
                          return (
                            <div key={idx} className="prose prose-sm md:prose-base dark:prose-invert max-w-none prose-p:leading-relaxed">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{section}</ReactMarkdown>
                            </div>
                          );
                        }

                        // It's a structured ## Header section
                        const lines = section.trim().split("\n");
                        const headerLine = lines[0].replace("##", "").trim();
                        const bodyContent = lines.slice(1).join("\n").trim();
                        if (!bodyContent && !isLoading) return null; // hide empty sections unless still typing

                        let Icon = Brain;
                        if (headerLine.toLowerCase().includes("summary")) Icon = Activity;
                        if (headerLine.toLowerCase().includes("metrics")) Icon = BarChart2;

                        return (
                          <div key={idx} className="bg-white dark:bg-white/5 rounded-xl border border-black/5 dark:border-white/10 shadow-sm overflow-hidden">
                            <div className="p-4 bg-gray-50/50 dark:bg-white/[0.02] border-b border-black/5 dark:border-white/10 flex items-center gap-2">
                              <Icon className="w-5 h-5 text-indigo-500" />
                              <span className="font-semibold text-gray-800 dark:text-gray-200">{headerLine}</span>
                            </div>
                            <div className="p-5 prose prose-sm md:prose-base dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-black/5 dark:prose-pre:bg-white/5 prose-pre:border prose-pre:border-black/5 dark:prose-pre:border-white/5 prose-li:my-0 pb-1">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{bodyContent}</ReactMarkdown>
                            </div>
                            {headerLine.toLowerCase().includes("metrics") && (
                              <div className="px-5 pb-5">
                                <button
                                  ref={metricsBtnRef}
                                  onClick={() => setIsMetricsModalOpen(true)}
                                  className="mt-3 flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500/10 to-cyan-500/10 border border-indigo-500/20 hover:border-indigo-500/40 text-indigo-600 dark:text-indigo-400 text-sm font-medium transition-all hover:scale-[1.02] active:scale-[0.98] group"
                                >
                                  <FlaskConical className="w-4 h-4 group-hover:rotate-12 transition-transform duration-300" />
                                  View Performance Calculations
                                </button>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ) : msg.role === "user" ? (
                    <div className="text-[15px] leading-relaxed whitespace-pre-wrap">
                      {msg.content}
                    </div>
                  ) : (
                    <div className="flex items-center gap-1 h-6">
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                  )}
                </SquircleCard>
              </div>
            ))
          )}
          <div className="h-[40vh] shrink-0 pointer-events-none" />
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area (z-index 100 prevents SquircleCards from scrolling over and hiding the input box) */}
        <div className="absolute bottom-0 left-0 right-0 z-[100] p-4 md:p-6 bg-gradient-to-t from-white via-white to-transparent dark:from-black dark:via-black pb-8 pointer-events-none">
          <div className="max-w-4xl mx-auto relative group pointer-events-auto">
            <div className="absolute -inset-1 bg-gradient-to-r from-cyan-400 to-indigo-500 rounded-[28px] blur opacity-20 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
            <form
              onSubmit={handleSubmit}
              className="relative flex flex-col bg-white dark:bg-[#1a1a1a] p-2 rounded-[24px] border border-black/5 dark:border-white/10 shadow-lg transition-all"
            >
              {/* Premium Image/File Preview Area */}
              {selectedFile && (
                <div className="px-3 pt-3 pb-1">
                  <div className="relative inline-flex items-center gap-3 bg-gray-50 dark:bg-black/40 pr-4 pl-2 py-2 rounded-2xl border border-black/5 dark:border-white/10 group/file">
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center shrink-0 overflow-hidden shadow-inner">
                       {selectedFile.type.startsWith("image/") ? (
                         // eslint-disable-next-line @next/next/no-img-element
                         <img src={URL.createObjectURL(selectedFile)} alt="preview" className="w-full h-full object-cover" />
                       ) : (
                         <Paperclip className="w-5 h-5 text-indigo-500" />
                       )}
                    </div>
                    <div className="flex flex-col max-w-[200px]">
                      <span className="text-sm font-semibold text-gray-800 dark:text-gray-200 truncate">{selectedFile.name}</span>
                      <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </div>
                    <button 
                      type="button" 
                      onClick={() => setSelectedFile(null)}
                      className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-md opacity-0 group-hover/file:opacity-100 transition-opacity hover:scale-110"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              )}
              
              <div className="relative flex items-end gap-2 w-full">
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={(e) => { if(e.target.files) setSelectedFile(e.target.files[0]) }} 
                  className="hidden" 
                  accept="image/*,application/pdf" 
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="p-3 text-gray-400 hover:text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-500/10 rounded-2xl transition-all shrink-0 mb-1 ml-1"
                >
                  <Paperclip className="w-5 h-5" />
                </button>
                <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (input.trim() && !isLoading) handleSubmit(e);
                  }
                }}
                placeholder="Describe a clinical case..."
                className="w-full max-h-48 min-h-[52px] px-4 py-3 bg-transparent border-none focus:ring-0 resize-none text-[15px] placeholder:text-gray-400"
                rows={1}
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={(!input.trim() && !selectedFile) || isLoading}
                className="p-3 bg-black dark:bg-white text-white dark:text-black rounded-2xl hover:scale-[1.05] hover:-translate-y-0.5 disabled:opacity-50 disabled:hover:scale-100 disabled:hover:translate-y-0 transition-all shadow-md shrink-0 mb-1 mr-1"
              >
                <Send className="w-5 h-5" />
              </button>
              </div>
            </form>
            <div className="text-center mt-3 text-xs text-gray-400">
              Clinsight can make mistakes. Verify clinical recommendations.
            </div>
          </div>
        </div>

      </div>

      {/* Real-time Architecture Visualizer Modal */}
      <ArchitectureModal
        isOpen={isWorkflowModalOpen}
        onClose={() => setIsWorkflowModalOpen(false)}
        activeTraces={messages[messages.length - 1]?.traces || []}
        contentString={messages[messages.length - 1]?.content || ""}
        hasCases={messages.length > 0 && Array.isArray(messages[messages.length - 1]?.cases) && (messages[messages.length - 1]?.cases?.length ?? 0) > 0}
      />

      {/* Performance Calculations Modal with Genie Effect */}
      <GenieModal
        isOpen={isMetricsModalOpen}
        onClose={() => setIsMetricsModalOpen(false)}
        triggerRef={metricsBtnRef}
        title="Performance Metrics — Calculation Methodology"
      >
        <MetricsCalculationsContent />
      </GenieModal>
    </main>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Standalone content for the Performance Calculations Modal
// ──────────────────────────────────────────────────────────────────────────
function MetricsCalculationsContent() {
  const sections = [
    {
      title: "Task: Patient-to-Patient Retrieval (PPR)",
      color: "indigo",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
            Our system implements the <strong>PPR task</strong> from the PMC-Patients benchmark paper. Given a query patient, we retrieve the{" "}
            <em>k</em> most clinically similar patients from a corpus of <strong>250,294 cases</strong>.
          </p>
          <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800 text-xs text-amber-800 dark:text-amber-300">
            ⚠️ The current runtime metrics are <strong>placeholder estimates</strong> computed from Cross-Encoder logits, not against the PMC-Patients gold ground truth. The correct evaluation uses the{" "}
            <code className="font-mono bg-amber-100 dark:bg-amber-900/40 px-1 rounded">similar_patients</code> dictionary (planned for the Future Evaluation Phase).
          </div>
        </>
      ),
    },
    {
      title: "Ground Truth",
      color: "cyan",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
            The PMC-Patients dataset provides for each patient a <code className="font-mono text-xs bg-gray-100 dark:bg-white/10 px-1 rounded">similar_patients</code> dictionary:
          </p>
          <div className="mt-2 font-mono text-xs bg-gray-900 text-green-400 rounded-xl p-4 overflow-x-auto">
            {`similar_patients: { "patient_uid_abc": 2, "patient_uid_xyz": 1, ... }`}
          </div>
          <ul className="mt-3 text-sm text-gray-600 dark:text-gray-400 space-y-1">
            <li>• rel = <strong>2</strong>: Highly similar case</li>
            <li>• rel = <strong>1</strong>: Moderately similar case</li>
            <li>• rel = <strong>0</strong>: Not in similar_patients → irrelevant</li>
          </ul>
        </>
      ),
    },
    {
      title: "Recall@K",
      color: "green",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400">"Of all relevant patients, how many did we find in the top K?"</p>
          <div className="mt-3 bg-gray-900 text-cyan-300 font-mono text-sm rounded-xl p-4 leading-loose">
            Recall@K(q) = |R<sub>K</sub>(q) ∩ G(q)| / |G(q)|
          </div>
          <ul className="mt-3 text-xs text-gray-500 dark:text-gray-400 space-y-1">
            <li>• R_K(q) = top-K retrieved patients</li>
            <li>• G(q) = all patients in similar_patients for query q</li>
          </ul>
        </>
      ),
    },
    {
      title: "Precision@K",
      color: "blue",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400">"Of the K retrieved patients, how many are truly relevant?"</p>
          <div className="mt-3 bg-gray-900 text-cyan-300 font-mono text-sm rounded-xl p-4 leading-loose">
            Precision@K(q) = |R<sub>K</sub>(q) ∩ G(q)| / K
          </div>
        </>
      ),
    },
    {
      title: "NDCG@K (Normalized Discounted Cumulative Gain)",
      color: "purple",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400">"Does rank 1 have a rel=2 case? NDCG rewards putting high-relevance hits at the top."</p>
          <div className="mt-3 bg-gray-900 text-cyan-300 font-mono text-xs rounded-xl p-4 leading-loose space-y-1">
            <p>DCG@K = Σ (2^rel_i − 1) / log₂(i + 1)</p>
            <p>IDCG@K = DCG of ideal ranking (sorted by rel desc)</p>
            <p>NDCG@K = DCG@K / IDCG@K</p>
          </div>
          <p className="mt-3 text-xs text-gray-500">The Future Evaluation Phase will replace this with the true PMC-Patients <code className="font-mono">similar_patients</code> dictionary evaluated across 1,000+ queries.</p>
        </>
      ),
    },
    {
      title: "MAP@K (Mean Average Precision)",
      color: "orange",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400">"How precise is our ranked list on average, measured at each relevant hit?"</p>
          <div className="mt-3 bg-gray-900 text-cyan-300 font-mono text-xs rounded-xl p-4 leading-loose space-y-1">
            <p>AP@K(q) = (1/|G(q)|) × Σ Prec@i × 𝟙(d_i ∈ G(q))</p>
            <p>MAP@K = (1/|Q|) × Σ AP@K(q) over all queries Q</p>
          </div>
        </>
      ),
    },
    {
      title: "MRR (Mean Reciprocal Rank)",
      color: "pink",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400">"At what rank is the very first relevant patient retrieved?"</p>
          <div className="mt-3 bg-gray-900 text-cyan-300 font-mono text-sm rounded-xl p-4">
            MRR = (1/|Q|) × Σ 1 / rank_first_relevant(q)
          </div>
          <p className="mt-2 text-xs text-gray-500">If query q finds its first relevant result at rank 3, its contribution is 1/3 = 0.333.</p>
        </>
      ),
    },
    {
      title: "Complete Step-by-Step Mathematical Example (@10)",
      color: "red",
      content: (
        <>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            Assume a query retrieves 10 patients: <strong>R = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]</strong>.<br/>
            The ground truth (G) says: <strong>P1 (rel=2), P3 (rel=1), P7 (rel=2)</strong>.<br/>
            Total relevant cases in G: <strong>|G| = 3</strong>.
          </p>
          <div className="bg-gray-900 text-cyan-300 font-mono text-xs rounded-xl p-4 leading-loose overflow-x-auto">
            <p className="text-gray-400">{"// 1. Calculate Recall@10"}</p>
            <p className="text-green-400">Formula: (Relevant hits found in top 10) / (Total relevant in G)</p>
            <p>Hits found: P1, P3, and P7 (3 hits).</p>
            <p>Recall@10 = 3 / 3 = <strong>1.0000 (100.0%)</strong></p>
            
            <p className="mt-4 text-gray-400">{"// 2. Calculate Precision@10"}</p>
            <p className="text-green-400">Formula: (Relevant hits found) / (Total retrieved at K=10)</p>
            <p>Precision@10 = 3 / 10 = <strong>0.3000 (30.00%)</strong></p>
            
            <p className="mt-4 text-gray-400">{"// 3. Calculate NDCG@10"}</p>
            <p className="text-green-400">Formula: DCG / Ideal_DCG</p>
            <p>• P1 is at rank 1 (rel=2) → DCG_P1 = (2² - 1) / log₂(1+1) = 3.0 / 1.0 = 3.0</p>
            <p>• P3 is at rank 3 (rel=1) → DCG_P3 = (2¹ - 1) / log₂(3+1) = 1.0 / 2.0 = 0.5</p>
            <p>• P7 is at rank 7 (rel=2) → DCG_P7 = (2² - 1) / log₂(7+1) = 3.0 / 3.0 = 1.0</p>
            <p>Total DCG = 3.0 + 0.5 + 1.0 = <strong>4.5</strong></p>
            <p className="text-purple-300">Ideal DCG (if P1(rel=2) at rank 1, P7(rel=2) at rank 2, P3(rel=1) at rank 3):</p>
            <p>• Rank 1: (2² - 1)/log₂(2) = 3.0</p>
            <p>• Rank 2: (2² - 1)/log₂(3) = 3.0 / 1.585 = 1.893</p>
            <p>• Rank 3: (2¹ - 1)/log₂(4) = 1.0 / 2.0 = 0.5</p>
            <p>IDCG = 3.0 + 1.893 + 0.5 = <strong>5.393</strong></p>
            <p>NDCG@10 = 4.5 / 5.393 = <strong>0.8344</strong></p>

            <p className="mt-4 text-gray-400">{"// 4. Calculate MAP@10 (Mean Average Precision)"}</p>
            <p className="text-green-400">Formula: Sum of (Precision at each hit) / (Total relevant in G)</p>
            <p>• Hit 1 at rank 1 (P1): Precision = 1/1 = 1.0</p>
            <p>• Hit 2 at rank 3 (P3): Precision = 2/3 = 0.666</p>
            <p>• Hit 3 at rank 7 (P7): Precision = 3/7 = 0.428</p>
            <p>Sum = 2.094</p>
            <p>AP@10 = 2.094 / 3 (total rel) = <strong>0.6980</strong></p>

            <p className="mt-4 text-gray-400">{"// 5. Calculate MRR (Mean Reciprocal Rank)"}</p>
            <p className="text-green-400">Formula: 1 / (Rank of first relevant hit)</p>
            <p>First relevant hit is P1 at rank 1.</p>
            <p>MRR = 1 / 1 = <strong>1.0000</strong></p>
          </div>
        </>
      ),
    },
  ];

  const colorMap: Record<string, string> = {
    indigo: "from-indigo-500/10 border-indigo-200 dark:border-indigo-800/50",
    cyan: "from-cyan-500/10 border-cyan-200 dark:border-cyan-800/50",
    green: "from-green-500/10 border-green-200 dark:border-green-800/50",
    blue: "from-blue-500/10 border-blue-200 dark:border-blue-800/50",
    purple: "from-purple-500/10 border-purple-200 dark:border-purple-800/50",
    orange: "from-orange-500/10 border-orange-200 dark:border-orange-800/50",
    pink: "from-pink-500/10 border-pink-200 dark:border-pink-800/50",
    red: "from-red-500/10 border-red-200 dark:border-red-800/50",
  };

  const dotMap: Record<string, string> = {
    indigo: "bg-indigo-500", cyan: "bg-cyan-500", green: "bg-green-500",
    blue: "bg-blue-500", purple: "bg-purple-500", orange: "bg-orange-500",
    pink: "bg-pink-500", red: "bg-red-500",
  };

  return (
    <div className="space-y-5">
      <p className="text-xs text-gray-400 leading-relaxed">
        This modal documents the exact Information Retrieval (IR) methodology used to evaluate Clinsight against the PMC-Patients PPR benchmark.
        These formulas are identical to the calculation protocol published in the original research paper.
      </p>
      {sections.map((s, i) => (
        <div key={i} className={`rounded-2xl border bg-gradient-to-br ${colorMap[s.color]} to-transparent p-4`}>
          <div className="flex items-center gap-2 mb-3">
            <div className={`w-2 h-2 rounded-full ${dotMap[s.color]}`} />
            <h3 className="font-semibold text-sm text-gray-800 dark:text-gray-200">{s.title}</h3>
          </div>
          <div>{s.content}</div>
        </div>
      ))}
    </div>
  );
}
