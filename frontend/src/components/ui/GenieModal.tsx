"use client";

import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface GenieModalProps {
    isOpen: boolean;
    onClose: () => void;
    triggerRef?: React.RefObject<HTMLElement | null>; // Kept for prop compatibility
    children: React.ReactNode;
    title: string;
}

export function GenieModal({ isOpen, onClose, children, title }: GenieModalProps) {
    // Prevent body scrolling when modal is open
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }
        return () => { document.body.style.overflow = 'unset'; };
    }, [isOpen]);

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 z-[200] bg-black/40 backdrop-blur-sm"
                    />

                    {/* Modal Centered Container */}
                    <div className="fixed inset-0 z-[201] flex items-center justify-center p-4 md:p-8 pointer-events-none">
                        {/* Modal panel */}
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
                            className="w-full max-w-3xl max-h-[90vh] flex flex-col rounded-3xl bg-white dark:bg-[#0f0f0f] border border-black/10 dark:border-white/10 shadow-2xl overflow-hidden pointer-events-auto"
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between p-5 border-b border-black/5 dark:border-white/5 shrink-0 bg-gray-50/80 dark:bg-white/[0.03] backdrop-blur-lg">
                                <span className="font-semibold text-base tracking-tight">{title}</span>
                                <button
                                    onClick={onClose}
                                    className="w-8 h-8 rounded-full flex items-center justify-center bg-gray-100 dark:bg-white/10 hover:bg-gray-200 dark:hover:bg-white/20 transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>

                            {/* Scrollable content */}
                            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
                                {children}
                            </div>
                        </motion.div>
                    </div>
                </>
            )}
        </AnimatePresence>
    );
}
