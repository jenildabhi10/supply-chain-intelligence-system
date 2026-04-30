import { api, type RiskSummary } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { TierBadge } from "@/components/TierBadge";
import { MetricCard } from "@/components/MetricCard";
import { Card } from "@/components/Card";
import PortMapClient from "@/components/PortMapClient";
import Link from "next/link";
import { ArrowUpRight, AlertTriangle } from "lucide-react";
import { tierColor } from "@/lib/utils";

async function safeFetchAll() {
  try {
    const [ports, risk] = await Promise.all([api.ports(), api.allRisk()]);
    return { ports: ports.ports, summaries: risk.port_summaries, generatedAt: risk.generated_at, error: null as string | null };
  } catch (err) {
    return { ports: [], summaries: [], generatedAt: null, error: err instanceof Error ? err.message : "API unavailable" };
  }
}

export default async function OverviewPage() {
  const { ports, summaries, generatedAt, error } = await safeFetchAll();

  const tierCounts: Record<string, number> = { critical: 0, high: 0, medium: 0, low: 0 };
  summaries.forEach((s) => { tierCounts[s.risk_tier] = (tierCounts[s.risk_tier] ?? 0) + 1; });
  const avgScore = summaries.length
    ? summaries.reduce((acc, s) => acc + s.risk_score, 0) / summaries.length
    : 0;
  const totalSignals = summaries.reduce((acc, s) => acc + s.n_signals, 0);
  const topThree = summaries.slice().sort((a, b) => b.risk_score - a.risk_score).slice(0, 3);

  return (
    <div className="space-y-12">
      {/* Hero */}
      <section>
        <div className="font-mono text-xs text-fg-dim mb-3">{"// supply_chain_intelligence.system"}</div>
        <h1 className="text-3xl md:text-4xl font-medium tracking-tight text-fg">
          U.S. port disruption risk, in real time.
        </h1>
        <p className="mt-3 text-fg-muted max-w-2xl leading-relaxed">
          Evidence-grounded risk scores across {ports.length || 10} major U.S. ports — fused from
          weather, seismic, fire, news, and trade-flow signals. Backed by an XGBoost
          model and explained by a Groq-powered intelligence agent.
        </p>
        {error && (
          <div className="mt-4 inline-flex items-center gap-2 rounded-md border border-tier-critical/40 bg-tier-critical/10 px-3 py-2 text-xs font-mono text-tier-critical">
            <AlertTriangle size={12} /> API offline — {error}. Run: <code className="bg-bg px-1 rounded">uvicorn src.api.main:app</code>
          </div>
        )}
      </section>

      {/* KPI strip */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="avg risk score"
          value={avgScore.toFixed(1)}
          hint={`${summaries.length} ports monitored`}
          accent="var(--color-accent)"
        />
        <MetricCard
          label="critical / high"
          value={`${tierCounts.critical} / ${tierCounts.high}`}
          hint="ports needing action"
          accent="var(--color-tier-critical)"
        />
        <MetricCard
          label="medium"
          value={tierCounts.medium}
          hint="ports to watch"
          accent="var(--color-tier-medium)"
        />
        <MetricCard
          label="active signals"
          value={totalSignals}
          hint="last 7 days"
          accent="var(--color-tier-low)"
        />
      </section>

      {/* Map + top risks */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <SectionHeader num="01" title="risk landscape" hint="click a marker to inspect" />
          <PortMapClient ports={ports} summaries={summaries} />
        </div>
        <div>
          <SectionHeader num="02" title="top risks" hint="ranked by composite score" />
          <div className="space-y-3">
            {topThree.length === 0 && (
              <Card className="p-6 text-fg-dim font-mono text-xs">
                {"// no risk data available"}
              </Card>
            )}
            {topThree.map((s) => <TopRiskCard key={s.port_id} s={s} />)}
          </div>
        </div>
      </section>

      {/* Port table */}
      <section>
        <SectionHeader
          num="03"
          title="all monitored ports"
          hint="sortable · click to drill in"
          right={
            <Link
              href="/ports"
              className="font-mono text-[11px] text-accent hover:underline inline-flex items-center gap-1"
            >
              view full table <ArrowUpRight size={12} />
            </Link>
          }
        />
        <RiskRowList rows={summaries.slice(0, 6)} />
      </section>

      {generatedAt && (
        <div className="font-mono text-[11px] text-fg-dim">
          {"// last refresh: "}{new Date(generatedAt).toLocaleString()}
        </div>
      )}
    </div>
  );
}

function TopRiskCard({ s }: { s: RiskSummary }) {
  return (
    <Link
      href={`/ports/${s.port_id}`}
      className="block group"
    >
      <Card className="p-4 hover:border-border-strong transition" >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-fg group-hover:text-accent transition font-medium truncate">
              {s.port_name}
            </div>
            <div className="font-mono text-[10px] text-fg-dim">{s.port_id}</div>
          </div>
          <TierBadge tier={s.risk_tier} />
        </div>
        <div className="mt-3 flex items-baseline gap-2">
          <span
            className="text-2xl font-medium tabular-nums"
            style={{ color: tierColor(s.risk_tier) }}
          >
            {s.risk_score.toFixed(1)}
          </span>
          <span className="font-mono text-[11px] text-fg-dim">
            p={(s.p_disruption * 100).toFixed(0)}% · {s.n_signals} signals
          </span>
        </div>
        {s.top_driver && (
          <div className="mt-2 font-mono text-[11px] text-fg-muted truncate">
            ↳ {s.top_driver}
          </div>
        )}
      </Card>
    </Link>
  );
}

function RiskRowList({ rows }: { rows: RiskSummary[] }) {
  if (rows.length === 0) {
    return <Card className="p-6 text-fg-dim font-mono text-xs">{"// no rows to display"}</Card>;
  }
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={r.port_id}
              className={i === 0 ? "" : "border-t border-border"}
            >
              <td className="px-4 py-3 w-12 font-mono text-xs text-fg-dim">{String(i + 1).padStart(2, "0")}</td>
              <td className="px-2 py-3">
                <Link href={`/ports/${r.port_id}`} className="text-fg hover:text-accent transition font-medium">
                  {r.port_name}
                </Link>
              </td>
              <td className="px-2 py-3 font-mono text-xs text-fg-muted tabular-nums">
                {r.risk_score.toFixed(1)}
              </td>
              <td className="px-2 py-3"><TierBadge tier={r.risk_tier} /></td>
              <td className="px-4 py-3 font-mono text-[11px] text-fg-dim hidden md:table-cell">
                p={(r.p_disruption * 100).toFixed(0)}% · {r.n_signals} sig
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
