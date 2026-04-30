import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function tierColor(tier: string): string {
  switch (tier.toLowerCase()) {
    case "critical": return "var(--color-tier-critical)";
    case "high":     return "var(--color-tier-high)";
    case "medium":   return "var(--color-tier-medium)";
    case "low":
    default:         return "var(--color-tier-low)";
  }
}

export function tierClass(tier: string): string {
  switch (tier.toLowerCase()) {
    case "critical": return "text-[color:var(--color-tier-critical)] border-[color:var(--color-tier-critical)]/40 bg-[color:var(--color-tier-critical)]/10";
    case "high":     return "text-[color:var(--color-tier-high)] border-[color:var(--color-tier-high)]/40 bg-[color:var(--color-tier-high)]/10";
    case "medium":   return "text-[color:var(--color-tier-medium)] border-[color:var(--color-tier-medium)]/40 bg-[color:var(--color-tier-medium)]/10";
    case "low":
    default:         return "text-[color:var(--color-tier-low)] border-[color:var(--color-tier-low)]/40 bg-[color:var(--color-tier-low)]/10";
  }
}

export function formatPct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function formatScore(x: number): string {
  return x.toFixed(1);
}

export function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const diff = Math.max(0, now - then);
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
}
