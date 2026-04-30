"use client";

import { useState, useRef, useEffect, useTransition, FormEvent } from "react";
import { api } from "@/lib/api";
import { SendHorizontal, Sparkles, User, Loader2, Wrench } from "lucide-react";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  meta?: { tools: string[]; llm: boolean; model: string | null };
}

const SUGGESTIONS = [
  "Which port has the highest disruption risk right now?",
  "Why is la_lb at risk this week?",
  "Compare risk between ny_nj and savannah.",
  "What active weather signals affect houston?",
];

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isPending, startTransition] = useTransition();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isPending]);

  const send = (q: string) => {
    if (!q.trim() || isPending) return;
    const userMsg: Message = { role: "user", content: q };
    setMessages((m) => [...m, userMsg]);
    setInput("");

    startTransition(async () => {
      try {
        const r = await api.explain(q);
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: r.answer,
            meta: { tools: r.tool_calls_made, llm: r.llm_used, model: r.model },
          },
        ]);
      } catch (err) {
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: `// agent error: ${err instanceof Error ? err.message : "unknown"}`,
          },
        ]);
      }
    });
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-220px)] min-h-[480px] rounded-xl border border-border bg-bg-card overflow-hidden">
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-5">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-12 h-12 rounded-xl border border-border-strong bg-bg-elevated grid place-items-center mb-4">
              <Sparkles size={20} className="text-accent" />
            </div>
            <h3 className="text-fg font-medium">Ask the intelligence agent</h3>
            <p className="text-fg-muted text-sm mt-1 max-w-md">
              Grounded in real signals, model output, and SHAP drivers. Replies cite the tools used.
            </p>
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-2xl w-full">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-left text-sm px-3 py-2.5 rounded-lg border border-border bg-bg-elevated hover:border-accent/50 hover:bg-bg transition text-fg-muted hover:text-fg"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <Bubble key={i} m={m} />
        ))}
        {isPending && (
          <div className="flex gap-3 items-start">
            <Avatar role="assistant" />
            <div className="rounded-xl border border-border bg-bg-elevated px-4 py-3 flex items-center gap-2 font-mono text-xs text-fg-muted">
              <Loader2 size={12} className="animate-spin" />
              thinking…
            </div>
          </div>
        )}
      </div>

      <form
        onSubmit={onSubmit}
        className="border-t border-border bg-bg-elevated p-3 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about port risk, signals, or scenarios…"
          className="flex-1 bg-bg border border-border rounded-md px-4 py-2.5 text-sm text-fg placeholder:text-fg-dim focus:outline-none focus:border-accent transition"
        />
        <button
          type="submit"
          disabled={isPending || !input.trim()}
          className="px-4 rounded-md border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20 transition disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <SendHorizontal size={16} />
        </button>
      </form>
    </div>
  );
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  return (
    <div
      className={cn(
        "shrink-0 w-8 h-8 rounded-md grid place-items-center border",
        role === "assistant"
          ? "bg-accent/10 border-accent/30 text-accent"
          : "bg-bg-elevated border-border text-fg-muted",
      )}
    >
      {role === "assistant" ? <Sparkles size={14} /> : <User size={14} />}
    </div>
  );
}

function Bubble({ m }: { m: Message }) {
  const isUser = m.role === "user";
  return (
    <div className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      <Avatar role={m.role} />
      <div
        className={cn(
          "max-w-[80%] rounded-xl border px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap",
          isUser
            ? "bg-bg-elevated border-border text-fg"
            : "bg-bg-card border-border-strong text-fg",
        )}
      >
        {m.content}
        {m.meta && (
          <div className="mt-2 pt-2 border-t border-border flex flex-wrap gap-2 font-mono text-[10px] text-fg-dim">
            {m.meta.tools.length > 0 && (
              <span className="inline-flex items-center gap-1">
                <Wrench size={10} /> {m.meta.tools.join(", ")}
              </span>
            )}
            {m.meta.llm && m.meta.model && (
              <span>· model: {m.meta.model}</span>
            )}
            {!m.meta.llm && <span>· fallback (no LLM)</span>}
          </div>
        )}
      </div>
    </div>
  );
}
