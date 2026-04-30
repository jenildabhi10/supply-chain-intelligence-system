import { api, type RiskCard, type AlertsResponse } from "@/lib/api";
import { SectionHeader } from "@/components/SectionHeader";
import { TierBadge } from "@/components/TierBadge";
import { RiskGauge } from "@/components/RiskGauge";
import { SHAPChart } from "@/components/SHAPChart";
import { ActiveSignals } from "@/components/ActiveSignals";
import { Simulator } from "@/components/Simulator";
import { MetricCard } from "@/components/MetricCard";
import { Card } from "@/components/Card";
import Link from "next/link";
import { ArrowLeft, AlertTriangle, ShieldAlert } from "lucide-react";
import { notFound } from "next/navigation";

interface PageProps {
  params: Promise<{ id: string }>;
}

async function safeFetch(id: string): Promise<{ card: RiskCard | null; alerts: AlertsResponse | null; error: string | null }> {
  try {
    const [card, alerts] = await Promise.all([
      api.portRisk(id),
      api.alerts(id, 14).catch(() => null),
    ]);
    return { card, alerts, error: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "fetch failed";
    if (msg.includes("404")) return { card: null, alerts: null, error: "404" };
    return { card: null, alerts: null, error: msg };
  }
}

export default async function PortDetailPage({ params }: PageProps) {
  const { id } = await params;
  const { card, alerts, error } = await safeFetch(id);

  if (error === "404") notFound();

  if (!card) {
    return (
      <div className="space-y-6">
        <Link href="/ports" className="inline-flex items-center gap-2 font-mono text-xs text-fg-muted hover:text-accent">
          <ArrowLeft size={12} /> back to ports
        </Link>
        <Card className="p-6 border-tier-critical/40">
          <div className="flex items-center gap-2 text-tier-critical font-mono text-xs">
            <AlertTriangle size={14} /> Could not load port {id} — {error}
          </div>
        </Card>
      </div>
    );
  }

  const signals = alerts?.signals ?? card.active_signals;

  return (
    <div className="space-y-10">
      {/* Header */}
      <section className="space-y-3">
        <Link href="/ports" className="inline-flex items-center gap-2 font-mono text-xs text-fg-muted hover:text-accent transition">
          <ArrowLeft size={12} /> back to ports
        </Link>
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <div className="font-mono text-xs text-fg-dim">{"// port: "}{card.port_id}</div>
            <h1 className="mt-1 text-3xl md:text-4xl font-medium tracking-tight text-fg">
              {card.port_name}
            </h1>
            <div className="mt-1 font-mono text-xs text-fg-muted">{card.region}</div>
          </div>
          <TierBadge tier={card.risk_tier} size="md" />
        </div>
      </section>

      {/* Top: gauge + KPIs */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1 p-6 flex items-center justify-center">
          <RiskGauge score={card.risk_score} tier={card.risk_tier} label="composite risk" />
        </Card>

        <div className="lg:col-span-2 grid grid-cols-2 gap-4">
          <MetricCard
            label="p(disruption)"
            value={`${(card.p_disruption * 100).toFixed(1)}%`}
            hint="model probability"
            accent="var(--color-tier-critical)"
          />
          <MetricCard
            label="anomaly score"
            value={card.anomaly_score.toFixed(2)}
            hint="isolation forest"
            accent="var(--color-tier-medium)"
          />
          <MetricCard
            label="impact index"
            value={card.impact_index.toFixed(2)}
            hint="trade exposure weight"
            accent="var(--color-accent)"
          />
          <MetricCard
            label="confidence"
            value={`${card.confidence_score.toFixed(0)}`}
            hint={`data completeness ${(card.data_completeness * 100).toFixed(0)}%`}
            accent="var(--color-tier-low)"
          />
        </div>
      </section>

      {/* Confidence flags */}
      {card.confidence_flags.length > 0 && (
        <section>
          <Card className="p-4 border-tier-medium/30">
            <div className="flex items-start gap-3">
              <ShieldAlert size={16} className="text-tier-medium shrink-0 mt-0.5" />
              <div className="min-w-0">
                <div className="font-mono text-xs text-tier-medium uppercase tracking-wider">
                  confidence advisories
                </div>
                <ul className="mt-1.5 space-y-0.5 text-xs text-fg-muted">
                  {card.confidence_flags.map((f, i) => (
                    <li key={i} className="font-mono">{"// "}{f}</li>
                  ))}
                </ul>
              </div>
            </div>
          </Card>
        </section>
      )}

      {/* SHAP + Signals */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <SectionHeader num="01" title="model drivers" hint="SHAP feature contributions" />
          <Card className="p-5">
            <SHAPChart drivers={card.top_shap_drivers} />
            <div className="mt-3 flex items-center gap-4 font-mono text-[10px] text-fg-dim">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm bg-tier-critical" /> increases risk
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm bg-tier-low" /> decreases risk
              </span>
            </div>
          </Card>
        </div>
        <div>
          <SectionHeader
            num="02"
            title="active signals"
            hint={`${signals.length} live · last 14 days`}
          />
          <Card className="p-4">
            <ActiveSignals signals={signals} />
          </Card>
        </div>
      </section>

      {/* Simulator + Actions */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <SectionHeader num="03" title="what-if simulator" hint="adjust inputs to test scenarios" />
          <Simulator
            portId={card.port_id}
            initialP={card.p_disruption}
            initialA={card.anomaly_score}
            baseline={card.risk_score}
          />
        </div>
        <div>
          <SectionHeader num="04" title="recommended actions" hint={`for ${card.risk_tier} tier`} />
          <Card className="p-5">
            <ul className="space-y-3">
              {card.recommended_actions.map((a, i) => (
                <li key={i} className="flex gap-3">
                  <span className="font-mono text-xs text-accent shrink-0 pt-0.5">{String(i + 1).padStart(2, "0")}</span>
                  <span className="text-sm text-fg-muted leading-relaxed">{a}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </section>

      {/* Footer meta */}
      <section className="pt-6 border-t border-border space-y-1 font-mono text-[11px] text-fg-dim">
        {card.model_info && (
          <div>
            {"// model: "}{card.model_info.version}
            {card.model_info.cv_pr_auc !== null && ` · CV PR-AUC ${card.model_info.cv_pr_auc.toFixed(3)}`}
            {card.model_info.n_training_rows && ` · ${card.model_info.n_training_rows} training rows`}
          </div>
        )}
        <div>{"// generated: "}{new Date(card.generated_at).toLocaleString()}</div>
        <div>{"// horizon: "}{card.forecast_horizon_days} days</div>
        {card.features_week_ending && (
          <div>{"// features as of: "}{card.features_week_ending}</div>
        )}
      </section>
    </div>
  );
}
