import { useState, useEffect, RefObject } from 'react';

/**
 * A hook that calculates the distance between the mouse cursor and a target element.
 * Useful for macOS Dock-style magnification and proximity effects.
 */
export function useMouseProximity(ref: RefObject<HTMLElement | null>, maxDistance: number = 200) {
    const [proximity, setProximity] = useState(0); // 0 (far) to 1 (cursor is over center)

    useEffect(() => {
        const element = ref.current;
        if (!element) return;

        const handleMouseMove = (e: MouseEvent) => {
            const rect = element.getBoundingClientRect();

            // Calculate center of element
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            // Calculate distance from cursor to center
            const distanceX = e.clientX - centerX;
            const distanceY = e.clientY - centerY;
            const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);

            // Map distance to a 0-1 scale, where 1 is center and 0 is maxDistance
            // Using a Gaussian-style falloff for that smooth "bell curve" feel requested by the user
            if (distance < maxDistance) {
                // e^(-(x^2 / (2*variance)))
                const variance = (maxDistance / 3) ** 2; // 99% falloff by maxDistance
                const curve = Math.exp(-(distance ** 2) / (2 * variance));
                setProximity(curve);
            } else {
                setProximity(0);
            }
        };

        // Using window so we track even when mouse leaves the element
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, [ref, maxDistance]);

    return proximity;
}
