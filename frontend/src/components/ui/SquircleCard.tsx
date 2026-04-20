"use client";

import { useRef } from "react";
import { motion } from "framer-motion";
import { useMouseProximity } from "@/hooks/useMouseProximity";
import { cn } from "@/lib/utils";

interface SquircleCardProps {
    children: React.ReactNode;
    className?: string;
    dynamicScaling?: boolean;
}

export function SquircleCard({ children, className, dynamicScaling = false }: SquircleCardProps) {
    const cardRef = useRef<HTMLDivElement>(null);
    const proximity = useMouseProximity(cardRef, 300);

    // Map 0-1 proximity to a gentle scale factor (e.g. 1.0 to 1.05)
    // This creates the macOS dock floating/magnification effect 
    const scale = dynamicScaling ? 1 + proximity * 0.05 : 1;
    const zIndex = dynamicScaling ? Math.round(proximity * 10) : 1;

    return (
        <motion.div
            ref={cardRef}
            animate={{
                scale,
                zIndex,
                boxShadow: dynamicScaling && proximity > 0.1
                    ? "0 20px 40px rgba(0,0,0,0.1), 0 5px 10px rgba(0,0,0,0.05)"
                    : "0 4px 16px rgba(0, 0, 0, 0.06), 0 1px 4px rgba(0, 0, 0, 0.02)"
            }}
            transition={{
                type: "spring",
                stiffness: 300,
                damping: 20,
                mass: 1
            }}
            className={cn(
                "glass-panel squircle p-6 transition-colors duration-300",
                className
            )}
        >
            {children}
        </motion.div>
    );
}
