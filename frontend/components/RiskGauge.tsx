"use client";

import { motion } from "framer-motion";
import { tierColor } from "@/lib/utils";

interface Props {
  score: number;          // 0–100
  tier: string;
  delta?: number | null;
  label?: string;
  size?: number;          // pixel diameter
}

export function RiskGauge({ score, tier, delta = null, label, size = 220 }: Props) {
  const radius = (size - 24) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const color = tierColor(tier);

  return (
    <div className="flex flex-col items-center" style={{ width: size }}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--color-border-strong)"
            strokeWidth={10}
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={10}
            strokeLinecap="round"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference * (1 - pct) }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ strokeDasharray: circumference }}
          />
        </svg>
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center">
            <div className="text-4xl font-medium tabular-nums" style={{ color }}>
              {score.toFixed(1)}
            </div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-fg-muted mt-1">
              {tier} risk
            </div>
            {delta !== null && delta !== undefined && (
              <div className={`mt-1 font-mono text-[11px] ${delta > 0 ? "text-tier-critical" : delta < 0 ? "text-tier-low" : "text-fg-dim"}`}>
                {delta > 0 ? "▲" : delta < 0 ? "▼" : "■"} {Math.abs(delta).toFixed(1)}
              </div>
            )}
          </div>
        </div>
      </div>
      {label && (
        <div className="mt-3 font-mono text-[11px] text-fg-dim text-center">
          {label}
        </div>
      )}
    </div>
  );
}
