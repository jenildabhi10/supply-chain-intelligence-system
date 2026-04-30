import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  hint?: string;
  accent?: string; // CSS color
  className?: string;
}

export function MetricCard({ label, value, hint, accent, className }: Props) {
  return (
    <div
      className={cn(
        "relative rounded-xl border border-border bg-bg-card p-5 overflow-hidden",
        className,
      )}
    >
      {accent && (
        <span
          className="absolute top-0 left-0 right-0 h-[2px]"
          style={{ background: accent }}
        />
      )}
      <div className="font-mono text-[11px] text-fg-dim uppercase tracking-wider">
        {label}
      </div>
      <div className="mt-2 text-2xl md:text-3xl font-medium text-fg tabular-nums">
        {value}
      </div>
      {hint && (
        <div className="mt-1 font-mono text-[11px] text-fg-muted">{hint}</div>
      )}
    </div>
  );
}
