"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { RiskSummary } from "@/lib/api";
import { TierBadge } from "./TierBadge";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

type SortKey = "risk_score" | "p_disruption" | "anomaly_score" | "confidence_score" | "n_signals";

const cols: { key: SortKey; label: string; format: (v: number) => string }[] = [
  { key: "risk_score",       label: "score",      format: (v) => v.toFixed(1) },
  { key: "p_disruption",     label: "p(disrupt)", format: (v) => `${(v * 100).toFixed(0)}%` },
  { key: "anomaly_score",    label: "anomaly",    format: (v) => v.toFixed(2) },
  { key: "confidence_score", label: "conf",       format: (v) => v.toFixed(0) },
  { key: "n_signals",        label: "signals",    format: (v) => `${v}` },
];

export function RiskTable({ rows }: { rows: RiskSummary[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("risk_score");
  const [dir, setDir] = useState<"asc" | "desc">("desc");

  const sorted = useMemo(() => {
    const out = rows.slice();
    out.sort((a, b) => {
      const av = a[sortKey] as number;
      const bv = b[sortKey] as number;
      return dir === "asc" ? av - bv : bv - av;
    });
    return out;
  }, [rows, sortKey, dir]);

  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setDir(dir === "asc" ? "desc" : "asc");
    else { setSortKey(k); setDir("desc"); }
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-bg-elevated text-left">
            <th className="font-mono text-[10px] uppercase tracking-wider text-fg-dim px-4 py-3">port</th>
            {cols.map((c) => (
              <th key={c.key} className="font-mono text-[10px] uppercase tracking-wider text-fg-dim px-3 py-3">
                <button
                  onClick={() => toggleSort(c.key)}
                  className="inline-flex items-center gap-1 hover:text-fg transition"
                >
                  {c.label}
                  {sortKey === c.key
                    ? dir === "asc"
                      ? <ArrowUp size={10} />
                      : <ArrowDown size={10} />
                    : <ArrowUpDown size={10} className="opacity-40" />}
                </button>
              </th>
            ))}
            <th className="font-mono text-[10px] uppercase tracking-wider text-fg-dim px-3 py-3">tier</th>
            <th className="font-mono text-[10px] uppercase tracking-wider text-fg-dim px-3 py-3">top driver</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr
              key={r.port_id}
              className={cn(
                "border-t border-border hover:bg-bg-elevated transition group",
                i % 2 === 0 ? "" : "bg-bg-card/40",
              )}
            >
              <td className="px-4 py-3">
                <Link
                  href={`/ports/${r.port_id}`}
                  className="text-fg group-hover:text-accent transition font-medium"
                >
                  {r.port_name}
                </Link>
                <div className="font-mono text-[10px] text-fg-dim">{r.port_id}</div>
              </td>
              {cols.map((c) => (
                <td key={c.key} className="px-3 py-3 font-mono text-[12px] text-fg-muted tabular-nums">
                  {c.format(r[c.key] as number)}
                </td>
              ))}
              <td className="px-3 py-3"><TierBadge tier={r.risk_tier} /></td>
              <td className="px-3 py-3 font-mono text-[11px] text-fg-muted truncate max-w-[180px]">
                {r.top_driver ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
