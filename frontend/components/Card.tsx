import { cn } from "@/lib/utils";
import type { HTMLAttributes, ReactNode } from "react";

interface Props extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  glow?: boolean;
}

export function Card({ children, className, glow, ...rest }: Props) {
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-bg-card",
        glow && "ring-1 ring-accent/30 shadow-[0_0_40px_-12px_rgba(110,168,255,0.35)]",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
