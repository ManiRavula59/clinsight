import { X, User, Search, Database, Layers, Brain, CheckCircle2, ShieldCheck, Microscope, Network, Merge, Crosshair, HeartPulse, ThumbsUp, BookOpen, Activity, BarChart2, Lightbulb } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

type ArchitectureModalProps = {
    isOpen: boolean;
    onClose: () => void;
    // We pass the active trace log to dynamically highlight the current stage
    activeTraces: string[];
    contentString?: string;
    hasCases?: boolean;
};

export const ArchitectureModal = ({ isOpen, onClose, activeTraces, contentString = "", hasCases = false }: ArchitectureModalProps) => {
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted || !isOpen) return null;

    // Determine active stages by scanning the trace logs
    const tracesText = activeTraces.join(" ");

    const isInputActive = true;
    const isPIIActive = tracesText.includes("Checking for PII");
    const isNERActive = tracesText.includes("Extracting clinical NER") || tracesText.includes("Querying FAISS");
    const lowerContent = contentString.toLowerCase();
    const isDatasetActive = isNERActive;
    const isHybridActive = tracesText.includes("Querying FAISS") || tracesText.includes("Sparse BM25") || tracesText.includes("E1:");
    const isColbertActive = tracesText.includes("E1:") || tracesText.includes("E2:");
    const isCrossEncoderActive = tracesText.includes("E2:") || tracesText.includes("F1:");
    const isKGActive = tracesText.includes("F1:") || tracesText.includes("Model Reasoning");
    const isCasesRetrievedActive = hasCases || tracesText.includes("Model Reasoning");
    const isAnalysisActive = lowerContent.includes("summary") || lowerContent.includes("analysis");
    const isAIThinkingActive = lowerContent.includes("thinking") || lowerContent.includes("diagnosis");
    const isMetricsActive = lowerContent.includes("metrics") || lowerContent.includes("confidence");
    const isComplete = tracesText.includes("Reasoning complete") && lowerContent.length > 500 && isMetricsActive;

    const PipelineNode = ({ title, desc, icon: Icon, isActive, isDone }: any) => {
        return (
            <div className={cn(
                "relative flex flex-col items-center p-4 rounded-xl border-2 transition-all duration-500 w-44 text-center",
                isActive && !isDone ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-950/30 shadow-[0_0_20px_rgba(6,182,212,0.3)] scale-105" :
                    isDone ? "border-green-500/50 bg-green-50/50 dark:bg-green-950/20" :
                        "border-gray-200 dark:border-white/10 bg-white dark:bg-white/5 opacity-50"
            )}>
                {isActive && !isDone && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-500 rounded-full animate-ping" />
                )}
                {isDone && (
                    <div className="absolute -top-2 -right-2">
                        <CheckCircle2 className="w-5 h-5 text-green-500 bg-white dark:bg-black rounded-full" />
                    </div>
                )}
                <div className={cn(
                    "w-12 h-12 rounded-full flex items-center justify-center mb-3 transition-colors duration-500",
                    isActive && !isDone ? "bg-cyan-100 text-cyan-600 dark:bg-cyan-900/50 dark:text-cyan-400" :
                        isDone ? "bg-green-100 text-green-600 dark:bg-green-900/50 dark:text-green-400" :
                            "bg-gray-100 text-gray-500 dark:bg-white/10 dark:text-gray-400"
                )}>
                    <Icon className="w-6 h-6" />
                </div>
                <h3 className={cn("font-bold text-[13px] mb-1", isActive ? "text-gray-900 dark:text-white" : "text-gray-500 dark:text-gray-400")}>{title}</h3>
                <p className="text-[11px] text-gray-500 dark:text-gray-400 leading-snug">{desc}</p>
            </div>
        );
    };

    const Arrow = ({ active }: { active: boolean }) => (
        <div className="flex-1 h-0.5 bg-gray-200 dark:bg-white/10 relative min-w-[30px]">
            <div className={cn(
                "absolute top-0 left-0 h-full bg-cyan-500 transition-all duration-1000",
                active ? "w-full" : "w-0"
            )} />
        </div>
    );

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            <div className="relative bg-[#fafafa] dark:bg-[#0a0a0a] w-full max-w-5xl rounded-3xl shadow-2xl overflow-hidden border border-black/10 dark:border-white/10 flex flex-col max-h-[90vh]">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-black/5 dark:border-white/5 bg-white dark:bg-black relative z-10">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white flex items-center gap-3">
                            <Network className="w-6 h-6 text-indigo-500" />
                            Clinsight System Architecture
                        </h2>
                        <p className="text-gray-500 dark:text-gray-400 mt-1">
                            Live visualization of the multi-stage medical retrieval and reasoning pipeline.
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-white/10 transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-8 md:p-12 overflow-y-auto flex-1 relative custom-scrollbar">

                    {/* Decorative Grid Background */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

                    {/* Pipeline Container */}
                    <div className="relative flex flex-col gap-12 max-w-4xl mx-auto">

                        {/* Row 1: Input -> Privacy -> Query NLP */}
                        <div className="flex items-center justify-between gap-4">
                            <PipelineNode title="Clinical Input" desc="Raw unstructured patient presentation" icon={User} isActive={isInputActive} isDone={isPIIActive || isComplete} />
                            <Arrow active={isPIIActive || isComplete} />
                            <PipelineNode title="Privacy Shield" desc="Presidio PII De-identification" icon={ShieldCheck} isActive={isPIIActive} isDone={isNERActive || isComplete} />
                            <Arrow active={isNERActive || isComplete} />
                            <PipelineNode title="Query Expansion" desc="Spacy Medical NER & Normalization" icon={Microscope} isActive={isNERActive} isDone={isDatasetActive || isComplete} />
                        </div>

                        {/* Middle connector Down from Query Expansion */}
                        <div className="flex justify-end pr-[5.5rem]">
                            <div className={cn("w-0.5 h-12 bg-gray-200 dark:bg-white/10 relative", (isDatasetActive || isComplete) && "bg-cyan-500")} />
                        </div>

                        {/* Row 2: Dataset -> Hybrid RRF -> ColBERT -> Cross-Encoder */}
                        <div className="flex items-center justify-between gap-4 flex-row-reverse">
                            <PipelineNode title="PMC-Patients DB" desc="Offline 163k Evidence Dataset" icon={BookOpen} isActive={isDatasetActive} isDone={isHybridActive || isComplete} />
                            <Arrow active={isHybridActive || isComplete} />
                            <PipelineNode title="Hybrid RRF Retrieval" desc="FAISS Vector + BM25 Sparse Index" icon={Database} isActive={isHybridActive} isDone={isColbertActive || isComplete} />
                            <Arrow active={isColbertActive || isComplete} />
                            <PipelineNode title="ColBERT Bi-Encoder" desc="Lightweight Semantic Rerank (N=20)" icon={Merge} isActive={isColbertActive} isDone={isCrossEncoderActive || isComplete} />
                            <Arrow active={isCrossEncoderActive || isComplete} />
                            <PipelineNode title="Cross-Encoder" desc="ms-marco Precision Rerank (N=5)" icon={Crosshair} isActive={isCrossEncoderActive} isDone={isKGActive || isComplete} />
                        </div>

                        {/* Middle connector Down from Cross-Encoder */}
                        <div className="flex justify-start pl-[5.5rem]">
                            <div className={cn("w-0.5 h-12 bg-gray-200 dark:bg-white/10 relative", (isKGActive || isComplete) && "bg-cyan-500")} />
                        </div>

                        {/* Row 3: KG Validation -> Cases Retrieved -> LLM Analysis -> AI Thinking */}
                        <div className="flex items-center justify-between gap-4">
                            <PipelineNode title="KG Validation" desc="Entity & Disease Family Overlap Check" icon={HeartPulse} isActive={isKGActive} isDone={isCasesRetrievedActive || isComplete} />
                            <Arrow active={isCasesRetrievedActive || isComplete} />
                            <PipelineNode title="Cases Retrieved" desc="UI Accordion Render" icon={Activity} isActive={isCasesRetrievedActive} isDone={isAnalysisActive || isComplete} />
                            <Arrow active={isAnalysisActive || isComplete} />
                            <PipelineNode title="LLM Analysis" desc="UI Generation" icon={Brain} isActive={isAnalysisActive} isDone={isAIThinkingActive || isComplete} />
                            <Arrow active={isAIThinkingActive || isComplete} />
                            <PipelineNode title="AI Thinking" desc="UI Generation" icon={Lightbulb} isActive={isAIThinkingActive} isDone={isMetricsActive || isComplete} />
                        </div>

                        {/* Middle connector Down from AI Thinking */}
                        <div className="flex justify-end pr-[5.5rem]">
                            <div className={cn("w-0.5 h-12 bg-gray-200 dark:bg-white/10 relative", (isMetricsActive || isComplete) && "bg-cyan-500")} />
                        </div>

                        {/* Row 4: Performance Metrics */}
                        <div className="flex items-center justify-start pr-[4.5rem] flex-row-reverse">
                            <PipelineNode title="Performance Metrics" desc="UI Generation" icon={BarChart2} isActive={isMetricsActive} isDone={isComplete} />
                        </div>

                    </div>
                </div>
            </div>
        </div>
    );
};
