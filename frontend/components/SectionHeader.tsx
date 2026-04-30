import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  num: string;
  title: string;
  hint?: string;
  right?: ReactNode;
  className?: string;
}

export function SectionHeader({ num, title, hint, right, className }: Props) {
  return (
    <div className={cn("flex items-end justify-between gap-4 mb-5", className)}>
      <div className="flex items-baseline gap-3 min-w-0">
        <span className="font-mono text-xs text-fg-dim tracking-wider">{num} /</span>
        <h2 className="text-xl md:text-2xl font-medium tracking-tight text-fg truncate">
          {title}
        </h2>
        {hint && (
          <span className="hidden md:inline font-mono text-[11px] text-fg-dim truncate">
            {"// "}{hint}
          </span>
        )}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}
