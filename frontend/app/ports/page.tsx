import { api } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { RiskTable } from "@/components/RiskTable";
import { Card } from "@/components/Card";
import { AlertTriangle } from "lucide-react";

async function safeFetchRisk() {
  try {
    const r = await api.allRisk();
    return { rows: r.port_summaries, generatedAt: r.generated_at, error: null as string | null };
  } catch (err) {
    return { rows: [], generatedAt: null, error: err instanceof Error ? err.message : "API unavailable" };
  }
}

export default async function PortsPage() {
  const { rows, generatedAt, error } = await safeFetchRisk();

  return (
    <div className="space-y-8">
      <section>
        <div className="font-mono text-xs text-fg-dim mb-2">{"// route: /ports"}</div>
        <h1 className="text-3xl font-medium tracking-tight text-fg">All monitored ports</h1>
        <p className="mt-2 text-fg-muted max-w-2xl">
          Sortable view of every port in the system. Click any row for the full risk card,
          SHAP drivers, signals, and what-if simulator.
        </p>
      </section>

      {error && (
        <Card className="p-4 border-tier-critical/40">
          <div className="flex items-center gap-2 text-tier-critical font-mono text-xs">
            <AlertTriangle size={14} /> API offline — {error}
          </div>
        </Card>
      )}

      <section>
        <SectionHeader
          num="01"
          title="risk table"
          hint={`${rows.length} ports · sorted by risk score`}
        />
        {rows.length > 0 ? (
          <RiskTable rows={rows} />
        ) : (
          !error && <Card className="p-8 text-center text-fg-dim font-mono text-xs">{"// no port data"}</Card>
        )}
      </section>

      {generatedAt && (
        <div className="font-mono text-[11px] text-fg-dim">
          {"// last refresh: "}{new Date(generatedAt).toLocaleString()}
        </div>
      )}
    </div>
  );
}
