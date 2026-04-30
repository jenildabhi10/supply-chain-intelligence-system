"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Globe2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const links = [
  { num: "01", label: "overview", href: "/", icon: Globe2 },
  { num: "02", label: "ports",    href: "/ports", icon: Activity },
  { num: "03", label: "agent",    href: "/agent", icon: Sparkles },
];

export function TopNav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<{ ok: boolean; llm: boolean } | null>(null);

  useEffect(() => {
    api.health()
      .then(h => setHealth({ ok: h.model_loaded, llm: h.llm_available }))
      .catch(() => setHealth({ ok: false, llm: false }));
  }, []);

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-bg/80 backdrop-blur supports-[backdrop-filter]:bg-bg/60">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-7 h-7 rounded-md border border-border-strong bg-bg-card grid place-items-center font-mono text-xs text-accent group-hover:border-accent transition">
            SC
          </div>
          <span className="font-mono text-sm text-fg-muted group-hover:text-fg transition">
            <span className="text-fg">supply_chain_intel</span>
            <span className="text-fg-dim">.system</span>
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {links.map(({ num, label, href }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "px-3 py-1.5 rounded-md font-mono text-xs flex items-center gap-2 transition",
                  active
                    ? "bg-bg-card text-fg border border-border-strong"
                    : "text-fg-muted hover:text-fg hover:bg-bg-elevated border border-transparent",
                )}
              >
                <span className="text-fg-dim">{num}</span>
                <span>/</span>
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="text-fg-dim hidden sm:inline">api</span>
          <span
            className={cn(
              "w-2 h-2 rounded-full",
              health === null
                ? "bg-fg-dim"
                : health.ok
                  ? "bg-tier-low pulse-ring"
                  : "bg-tier-critical",
            )}
            title={health?.ok ? "API healthy" : "API offline"}
          />
          <span className="text-fg-muted">{health === null ? "…" : health.ok ? "live" : "offline"}</span>
          <span className="text-fg-dim mx-1">·</span>
          <span className="text-fg-dim">llm</span>
          <span
            className={cn(
              "w-2 h-2 rounded-full",
              health === null
                ? "bg-fg-dim"
                : health.llm
                  ? "bg-accent"
                  : "bg-fg-dim",
            )}
          />
        </div>
      </div>
    </header>
  );
}
