"use client";

import { useState, useTransition } from "react";
import { api, type SimulateResponse } from "@/lib/api";
import { TierBadge } from "./TierBadge";
import { Card } from "./Card";
import { Sliders, Loader2 } from "lucide-react";

interface Props {
  portId: string;
  initialP: number;
  initialA: number;
  baseline: number;
}

export function Simulator({ portId, initialP, initialA, baseline }: Props) {
  const [p, setP] = useState(initialP);
  const [a, setA] = useState(initialA);
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const run = () => {
    setError(null);
    startTransition(async () => {
      try {
        const r = await api.simulate(portId, p, a);
        setResult(r);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Simulation failed");
      }
    });
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sliders size={14} className="text-accent" />
          <span className="font-mono text-xs text-fg-muted uppercase tracking-wider">
            what-if simulator
          </span>
        </div>
        <span className="font-mono text-[10px] text-fg-dim">
          baseline: {baseline.toFixed(1)}
        </span>
      </div>

      <div className="space-y-5">
        <Slider
          label="p(disruption)"
          hint="calibrated model probability"
          value={p}
          onChange={setP}
          color="var(--color-tier-critical)"
        />
        <Slider
          label="anomaly score"
          hint="isolation forest output"
          value={a}
          onChange={setA}
          color="var(--color-tier-medium)"
        />
      </div>

      <button
        onClick={run}
        disabled={isPending}
        className="mt-5 w-full inline-flex items-center justify-center gap-2 rounded-md border border-border-strong bg-bg-elevated hover:bg-bg hover:border-accent transition px-4 py-2.5 font-mono text-xs text-fg disabled:opacity-50"
      >
        {isPending && <Loader2 size={12} className="animate-spin" />}
        run simulation
      </button>

      {error && (
        <div className="mt-4 text-xs text-tier-critical font-mono">
          {"// "}{error}
        </div>
      )}

      {result && (
        <div className="mt-5 pt-5 border-t border-border space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] text-fg-dim uppercase">simulated risk</span>
            <TierBadge tier={result.risk_tier} />
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl font-medium tabular-nums text-fg">
              {result.risk_score.toFixed(1)}
            </span>
            <span className={`font-mono text-xs ${result.delta_score > 0 ? "text-tier-critical" : result.delta_score < 0 ? "text-tier-low" : "text-fg-dim"}`}>
              {result.delta_score > 0 ? "+" : ""}{result.delta_score.toFixed(1)} vs baseline
            </span>
          </div>
          <div>
            <div className="font-mono text-[10px] text-fg-dim uppercase mb-2">recommended actions</div>
            <ul className="space-y-1.5">
              {result.recommended_actions.map((act, i) => (
                <li key={i} className="text-xs text-fg-muted flex gap-2">
                  <span className="text-accent font-mono">›</span>
                  <span>{act}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
}

function Slider({
  label, hint, value, onChange, color,
}: {
  label: string; hint: string; value: number; onChange: (v: number) => void; color: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <div>
          <span className="font-mono text-xs text-fg">{label}</span>
          <span className="ml-2 font-mono text-[10px] text-fg-dim">{"// "}{hint}</span>
        </div>
        <span className="font-mono text-xs tabular-nums" style={{ color }}>
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-current"
        style={{ color, accentColor: color }}
      />
    </div>
  );
}
