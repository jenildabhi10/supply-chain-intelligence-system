import type { ActiveSignal } from "@/lib/api";
import { tierColor, relativeTime } from "@/lib/utils";
import { AlertTriangle, Cloud, Activity, Flame, Newspaper, Ship } from "lucide-react";

const SOURCE_META: Record<string, { icon: typeof Cloud; label: string }> = {
  nws:   { icon: Cloud,         label: "NWS weather" },
  gdelt: { icon: Newspaper,     label: "GDELT news" },
  usgs:  { icon: Activity,      label: "USGS seismic" },
  firms: { icon: Flame,         label: "FIRMS fire" },
  bts:   { icon: Ship,          label: "BTS port data" },
};

export function ActiveSignals({ signals }: { signals: ActiveSignal[] }) {
  if (!signals || signals.length === 0) {
    return (
      <div className="text-fg-dim font-mono text-xs py-4 px-1">
        {"// no active signals — port appears nominal"}
      </div>
    );
  }
  return (
    <ul className="space-y-2">
      {signals.slice(0, 12).map((s) => {
        const meta = SOURCE_META[s.source] ?? { icon: AlertTriangle, label: s.source };
        const Icon = meta.icon;
        const color = tierColor(s.severity);
        return (
          <li
            key={s.signal_id}
            className="flex gap-3 rounded-lg border border-border bg-bg-elevated p-3 hover:border-border-strong transition"
            style={{ borderLeft: `3px solid ${color}` }}
          >
            <div
              className="shrink-0 w-8 h-8 rounded-md grid place-items-center"
              style={{ background: `${color}1f`, color }}
            >
              <Icon size={14} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2 text-[10px] font-mono uppercase tracking-wider text-fg-dim">
                <span>{meta.label}</span>
                <span>{relativeTime(s.observed_at)}</span>
              </div>
              <div className="mt-1 text-sm text-fg leading-snug">
                {s.description}
              </div>
              {s.evidence_url && (
                <a
                  href={s.evidence_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 inline-block font-mono text-[11px] text-accent hover:underline"
                >
                  source ↗
                </a>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
