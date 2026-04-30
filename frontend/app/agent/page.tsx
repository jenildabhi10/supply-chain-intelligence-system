import { Chat } from "@/components/Chat";
import { Card } from "@/components/Card";
import { Sparkles, Wrench, Database } from "lucide-react";

export default function AgentPage() {
  return (
    <div className="space-y-8">
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-end">
        <div className="lg:col-span-2">
          <div className="font-mono text-xs text-fg-dim mb-2">{"// route: /agent"}</div>
          <h1 className="text-3xl md:text-4xl font-medium tracking-tight text-fg flex items-center gap-3">
            <Sparkles size={28} className="text-accent" />
            Intelligence agent
          </h1>
          <p className="mt-2 text-fg-muted max-w-2xl leading-relaxed">
            Natural-language access to the risk model. The agent calls live tools (port lookup,
            risk cards, signal feeds) and returns evidence-grounded answers — never hallucinated.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Card className="p-3">
            <div className="flex items-center gap-2 font-mono text-[10px] text-fg-dim uppercase">
              <Wrench size={11} /> tools
            </div>
            <div className="mt-1 text-xl font-medium text-fg">5</div>
          </Card>
          <Card className="p-3">
            <div className="flex items-center gap-2 font-mono text-[10px] text-fg-dim uppercase">
              <Database size={11} /> backend
            </div>
            <div className="mt-1 text-xl font-medium text-fg">Groq</div>
          </Card>
        </div>
      </section>

      <Chat />

      <section className="font-mono text-[11px] text-fg-dim">
        <div>{"// model: llama-3.3-70b-versatile (Groq free tier, 100k tok/day)"}</div>
        <div>{"// tools: list_ports · get_risk_card · get_all_port_scores · get_active_signals · get_top_risk_ports"}</div>
        <div>{"// fallback: structured scorer summary if rate-limited"}</div>
      </section>
    </div>
  );
}
