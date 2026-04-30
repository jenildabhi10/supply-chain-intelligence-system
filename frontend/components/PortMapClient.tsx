"use client";

import dynamic from "next/dynamic";
import type { PortInfo, RiskSummary } from "@/lib/api";

const PortMap = dynamic(() => import("./PortMap"), {
  ssr: false,
  loading: () => (
    <div
      className="w-full rounded-xl border border-border bg-bg-elevated grid place-items-center text-fg-dim font-mono text-xs"
      style={{ height: 460 }}
    >
      loading map…
    </div>
  ),
});

export default function PortMapClient(props: {
  ports: PortInfo[];
  summaries: RiskSummary[];
  height?: number;
}) {
  return <PortMap {...props} />;
}
