"use client";

import { useEffect, useState } from "react";
import { motion, useSpring } from "framer-motion";

export function BackgroundDiffusion() {
    const [isMounted, setIsMounted] = useState(false);

    // Smooth springs for cursor position to creating the trailing effect
    const springConfig = { damping: 50, stiffness: 200, mass: 1.5 };
    const mouseX = useSpring(0, springConfig);
    const mouseY = useSpring(0, springConfig);

    useEffect(() => {
        setIsMounted(true);
        // Center it initially
        mouseX.set(window.innerWidth / 2);
        mouseY.set(window.innerHeight / 2);

        const handleMouseMove = (e: MouseEvent) => {
            mouseX.set(e.clientX);
            mouseY.set(e.clientY);
        };

        window.addEventListener("mousemove", handleMouseMove);
        return () => window.removeEventListener("mousemove", handleMouseMove);
    }, [mouseX, mouseY]);

    if (!isMounted) return null;

    return (
        <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
            <motion.div
                className="absolute w-[800px] h-[800px] rounded-full opacity-10 blur-[100px]"
                style={{
                    x: mouseX,
                    y: mouseY,
                    translateX: "-50%",
                    translateY: "-50%",
                    background: "radial-gradient(circle, rgba(100,200,255,1) 0%, rgba(200,150,255,0.8) 50%, rgba(255,255,255,0) 80%)",
                }}
            />
        </div>
    );
}
