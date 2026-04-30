"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts";
import type { SHAPDriver } from "@/lib/api";

interface Props {
  drivers: SHAPDriver[];
  height?: number;
}

export function SHAPChart({ drivers, height = 320 }: Props) {
  if (!drivers || drivers.length === 0) {
    return (
      <div className="text-fg-dim font-mono text-xs py-6">
        {"// no SHAP drivers available"}
      </div>
    );
  }
  const data = drivers
    .slice()
    .sort((a, b) => Math.abs(b.shap_impact) - Math.abs(a.shap_impact))
    .slice(0, 8)
    .map((d) => ({
      feature: d.feature.replace(/_/g, " "),
      impact: d.shap_impact,
      value: d.value,
      direction: d.direction,
    }))
    .reverse();

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          <XAxis
            type="number"
            stroke="var(--color-fg-dim)"
            tick={{ fill: "var(--color-fg-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "var(--color-border)" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature"
            stroke="var(--color-fg-dim)"
            tick={{ fill: "var(--color-fg-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "var(--color-border)" }}
            tickLine={false}
            width={140}
          />
          <Tooltip
            cursor={{ fill: "var(--color-bg-elevated)" }}
            contentStyle={{
              background: "var(--color-bg-card)",
              border: "1px solid var(--color-border-strong)",
              borderRadius: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--color-fg)",
            }}
            formatter={(val) => [Number(val).toFixed(4), "SHAP impact"]}
          />
          <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.impact >= 0 ? "var(--color-tier-critical)" : "var(--color-tier-low)"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
